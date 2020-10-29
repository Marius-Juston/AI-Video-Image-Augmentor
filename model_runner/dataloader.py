import collections
import os

import cv2
import numpy as np


# SOME CODE DUPLICATED FROM https://github.com/thunil/TecoGAN

def inference_data_loader(input_dir_LR, input_dir_len=-1, input_dir_HR=None):
    filedir = input_dir_LR
    downSP = False
    if (input_dir_LR is None) or (not os.path.exists(input_dir_LR)):
        if (input_dir_HR is None) or (not os.path.exists(input_dir_HR)):
            raise ValueError('Input directory not found')
        filedir = input_dir_HR
        downSP = True

    image_list_LR_temp = os.listdir(filedir)
    image_list_LR_temp = [_ for _ in image_list_LR_temp if _.endswith(".png")]
    image_list_LR_temp = sorted(image_list_LR_temp)  # first sort according to abc, then sort according to 123
    image_list_LR_temp.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
    if input_dir_len > 0:
        image_list_LR_temp = image_list_LR_temp[:input_dir_len]

    image_list_LR = [os.path.join(filedir, _) for _ in image_list_LR_temp]

    # Read in and preprocess the images
    def preprocess_test(name):
        im = cv2.imread(name, 3).astype(np.float32)[:, :, ::-1]

        if downSP:
            icol_blur = cv2.GaussianBlur(im, (0, 0), sigmaX=1.5)
            im = icol_blur[::4, ::4, ::]
        im = im / 255.0  # np.max(im)
        return im

    image_LR = [preprocess_test(_) for _ in image_list_LR]

    if True:  # a hard-coded symmetric padding
        image_list_LR = image_list_LR[5:0:-1] + image_list_LR
        image_LR = image_LR[5:0:-1] + image_LR

    Data = collections.namedtuple('Data', 'paths_LR, inputs')
    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )
