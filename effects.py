import os

import PIL
import torch
from torch.autograd import Variable
from torchvision import transforms  # , utils
import numpy as np
from PIL import Image, ImageColor, ImageFile, ImageFilter
from data_loader import RescaleT
from data_loader import ToTensorLab
from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB


class U2Effects:

    # Output image formats
    FORMATS = [
        'np',  # Numpy array
        'pil',  # PIL Image
    ]

    MODEL_NAMES = [
        'u2net',  # U2Net (173.6 MB) - slower, but more accurate [Default]
        'u2netp',  # U2NetP (4.7 MB) - faster, but less accurate
    ]

    def __init__(self, model_name='u2net', cuda_mode=True, output_format='np'):
        self.model_name = model_name
        self.cuda_mode = cuda_mode and torch.cuda.is_available()  # Fallback to CPU mode, if cuda is not available
        self.trans = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
        self.output_format = output_format

        # Validate
        if output_format not in self.FORMATS:
            raise AssertionError('Invalid "output_format"', 'Use "np" or "pil"')
        if model_name not in self.MODEL_NAMES:
            raise AssertionError('Invalid "model_name"', 'Use "u2net" or "u2netp"')

        if model_name == 'u2net':
            print("Model: U2NET (173.6 MB)")
            self.net = U2NET(3, 1)  # 173.6 MB
        elif model_name == 'u2netp':
            print("Model: U2NetP (4.7 MB)")
            self.net = U2NETP(3, 1)  # 4.7 MB

        # Load network
        model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '.pth')

        if cuda_mode:
            print("CUDA mode")
            self.net.load_state_dict(torch.load(model_dir))
            self.net.cuda()
        else:
            print("CPU mode")
            self.net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

        self.net.eval()

    def get_mask(self, image_np):
        """ Returns image mask as Numpy Array image"""
        sample = self.preprocess(image_np)
        inputs_test = sample['image'].unsqueeze(0)
        inputs_test = inputs_test.type(torch.FloatTensor)
        if self.cuda_mode:
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        # Predict
        d1 = self.net(inputs_test)[0]
        predict = d1[:, 0, :, :]
        del d1
        predict = self.norm_pred(predict)  # normalization
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        # Prepare image mask
        image_np_size = (image_np.shape[1], image_np.shape[0])
        mask_pil = Image.fromarray(predict_np * 255).convert('RGB')
        mask_pil = mask_pil.resize(image_np_size, resample=Image.BILINEAR)  # Resize mask to original image

        # return np.asarray(mask_pil)
        return self.postprocess(mask_pil)

    @staticmethod
    def _np2pil(image_np):
        """ Converts Numpy Array image to PIL Image """
        return Image.fromarray(np.uint8(image_np)).convert('RGB')

    def get_object(self, image, color="#000000"):
        """ Returns image without background """
        mask_np = self.get_mask(image)
        mask_pil = self._np2pil(mask_np)
        image_pil = self._np2pil(image)
        background_pil = Image.new(mode='RGB', size=image_pil.size, color=color)
        object_pil = Image.composite(image_pil, background_pil, mask_pil.convert("L"))
        return self.postprocess(object_pil)

    def blur_background(self, image, blur_radius=5):
        """ Returns image with blured background using Gausian Blur """
        mask_pil = self.get_mask(image)
        background_pil = image.copy().filter(ImageFilter.GaussianBlur(blur_radius))
        object_pil = Image.composite(image, background_pil, mask_pil.convert("L"))
        return self.postprocess(object_pil)

    def get_background(self, image_pil, color="#000000"):
        """ Return background without main object """
        mask_pil = self.get_mask(image_pil)
        background_pil = Image.new(mode='RGB', size=image_pil.size, color=color)
        background_pil = Image.composite(background_pil, image_pil, mask_pil.convert("L"))
        return self.postprocess(background_pil)

    @staticmethod
    def postprocess(image):
        # If image is PIL.Image, convert to Numpy Array
        if isinstance(image, PIL.Image.Image):
            image = np.asarray(image)
        return image

    # image = Numpy Array
    def preprocess(self, image):
        # If image is PIL.Image, convert to Numpy Array
        # if isinstance(image_cv, PIL.Image.Image):
        #     image_cv = np.array(image_cv)[:, :, ::-1].copy()  # PIL image => OpenCv image

        label_3 = np.zeros(image.shape)
        label = np.zeros(label_3.shape[0:2])

        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        return self.trans({
            'imidx': np.array([0]),
            'image': image,
            'label': label
        })

    @staticmethod
    def norm_pred(d):
        """ Normalize the predicted SOD probability map """
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn
