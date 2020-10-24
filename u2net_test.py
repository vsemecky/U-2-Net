import os
import glob
from pprint import pprint
import argparse
from shutil import copyfile

import progressbar

from skimage import io, transform
import torch
import torchvision
from termcolor import colored
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim
import numpy as np
from PIL import Image, ImageColor
import cv2

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


# Convert PIL image to OpenCv image
def pil2opencv(pil_image):
    return np.array(pil_image)[:, :, ::-1].copy()


# Get average color at (top-left, top-right, bottom-left, bottom-right)
def has_correct_background(img, expected_color="#ffffff"):
    expected_color = ImageColor.getrgb(expected_color)

    pixdata = img.load()
    (w, h) = img.size
    pixels = [
        pixdata[0, 0],          # North-west
        pixdata[w - 1, 0],      # North-east
        pixdata[0, h - 1],      # South-west
        pixdata[w - 1, h - 1],  # South-east
    ]

    # Compare pixels with `expected_color`
    for pixel in pixels:
        # print("pixel", pixel, "expected_color", expected_color)
        if pixel != expected_color:
            return False

    return True


def save_output(image_name, predict, config):
    output_path = config.result_dir + os.sep + os.path.basename(image_name)

    # Load original image
    original_pil = Image.open(image_name).convert('RGB')

    # Prepare image mask
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    mask_pil = Image.fromarray(predict_np * 255).convert('RGB')
    # Resize mask to original image
    mask_pil = mask_pil.resize(original_pil.size, resample=Image.BILINEAR)

    # Masked image
    background_pil = Image.new(mode="RGB", size=original_pil.size, color=config.background)
    masked_pil = Image.composite(original_pil, background_pil, mask_pil.convert("L"))

    # Save mask
    if config.save_mask:
        mask_pil.save(output_path + ".mask.jpg")

    # Save original image
    if config.save_original:
        original_pil.save(output_path)

    # Save masked image
    if config.save_masked:
        masked_pil.save(output_path + ".masked.jpg")

    # Save comparison image
    if config.save_compare:
        compare_image_cv2 = cv2.hconcat([
            pil2opencv(original_pil),
            pil2opencv(mask_pil),
            pil2opencv(masked_pil),
        ])
        cv2.imwrite(output_path + ".compare.jpg", compare_image_cv2)


def main(config):
    # --------- 1. get image path and name ---------
    model_name = config.model
    image_dir = config.input_dir
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '.pth')
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print("Found images:", len(img_name_list))
    print("Model path:", model_dir)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    # --------- 3. model define ---------
    if model_name == 'u2net':
        print("Load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        print("Load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    # CUDA mode: 1 = CUDA, 0 = CPU
    cuda_mode = 1 if torch.cuda.is_available() else 0

    # Load network
    if cuda_mode:
        print("CUDA mode")
        net.load_state_dict(torch.load(model_dir))  # GPU mode
        net.cuda()
    else:
        print("CPU mode")
        net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))  # CPU mode

    net.eval()
    print()

    # Create result folder, if not exists
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir, exist_ok=True)

    # --------- 4. inference for each image ---------
    results = enumerate(test_salobj_dataloader)
    for i_test, data_test in progressbar.progressbar(results, redirect_stdout=True, max_value=len(test_salobj_dataloader)):
        try:
            image_file = img_name_list[i_test]
            basename = os.path.basename(image_file)

            # Skip if image has already target background
            original_pil = Image.open(image_file).convert('RGB')
            if has_correct_background(original_pil, config.background):
                print(basename, colored("SKIPPING (background OK)", "yellow"))
                copyfile(image_file, config.result_dir + os.sep + os.path.basename(image_file))
                continue

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if cuda_mode:
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

            # normalization
            predict = d1[:, 0, :, :]
            predict = normPRED(predict)

            # save results
            save_output(image_file, predict, config=config)

            del d1, d2, d3, d4, d5, d6, d7

            print(colored(basename, "yellow"), colored("OK", "green"))
        except Exception as e:
            print(colored(img_name_list[i_test], "yellow"), colored("ERROR", "red"), e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dir', help='Folder with input images', required=True)
    parser.add_argument('--result-dir', help='Output folder', required=True)
    parser.add_argument('--model', help='Model name "u2net" or "u2netp" (default: %(default)s)', default='u2net')
    parser.add_argument('--background', help='Background color (default: %(default)s)', default='#ffffff')
    parser.add_argument('--save-mask', type=bool, default=False,
                        help='Model name "u2net" or "u2netp" (default: %(default)s)')
    parser.add_argument('--save-masked', type=bool, default=False,
                        help='Save image without background (default: %(default)s)')
    parser.add_argument('--save-original', type=bool, default=False,
                        help='Copy original image to results (default: %(default)s)')
    parser.add_argument('--save-compare', type=bool, default=False,
                        help='Save comparison image [original,masked,mask] (default: %(default)s)')
    config = parser.parse_args()

    print("\nCONFIG:")
    pprint(vars(config))

    del parser
    main(config)
