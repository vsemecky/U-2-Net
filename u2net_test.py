import os
from pprint import pprint

from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import cv2
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB
import argparse


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


# Convert PIL image to OpenCv image
def pil2opencv(pil_image):
    return np.array(pil_image)[:, :, ::-1].copy()


def save_output(image_name, predict, config):
    output_path = config.result_dir + os.sep + os.path.basename(image_name)

    # Load original image
    original_pil = Image.open(image_name)
    original_np = np.array(original_pil)

    # Prepare image mask
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    mask_pil = Image.fromarray(predict_np * 255).convert('RGB')
    # Resize mask to original image
    mask_pil = mask_pil.resize(original_pil.size, resample=Image.BILINEAR)
    mask_np = np.array(mask_pil)

    # Masked image
    mask_np = mask_np / 255
    masked_np = original_np * mask_np
    masked_pil = Image.fromarray(masked_np.astype(np.uint8))

    # Save mask
    if config.save_mask:
        mask_pil.save(output_path + ".mask.jpg")

    # Save original image
    if config.save_original:
        original_pil.save(output_path + ".original.jpg")

    # Save masked image
    if config.save_masked:
        masked_pil.save(output_path + ".masked.jpg")

    # Save comparison image
    if config.save_compare:
        compare_image_cv2 = cv2.hconcat([
            pil2opencv(original_pil),
            pil2opencv(masked_pil),
            pil2opencv(mask_pil),
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
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if model_name == 'u2net':
        print("Load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        print("Load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    # Load network
    if torch.cuda.is_available():
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
    counter = 1
    count = len(test_salobj_dataloader)
    for i_test, data_test in enumerate(test_salobj_dataloader):
      try:
        print(counter, "/", count, img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        predict = d1[:, 0, :, :]
        predict = normPRED(predict)

        # save results
        save_output(
            img_name_list[i_test],
            predict,
            config=config
        )

        del d1, d2, d3, d4, d5, d6, d7
        counter += 1
      except Exception as e:
        print("SAVE EXCEPTION!!!", e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dir', help='Folder with input images', required=True)
    parser.add_argument('--result-dir', help='Output folder', required=True)
    parser.add_argument('--model', help='Model name "u2net" or "u2netp" (default: %(default)s)', default='u2net')
    parser.add_argument('--save-mask', type=bool, default=False,
                        help='Model name "u2net" or "u2netp" (default: %(default)s)')
    parser.add_argument('--save-masked', type=bool, default=False,
                        help='Save image without background (default: %(default)s)')
    parser.add_argument('--save-original', type=bool, default=False,
                        help='Copy original image to results (default: %(default)s)')
    parser.add_argument('--save-compare', type=bool, default=False,
                        help='Save comparison image [original,masked,mask] (default: %(default)s)')
    config = parser.parse_args()
    print("config", config)
    del parser
    main(config)
