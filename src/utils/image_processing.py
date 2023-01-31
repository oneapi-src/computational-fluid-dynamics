# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This script includes utility functions
for processing input car boundary images
'''

import numpy as np
from PIL import Image, ImageOps

def add_padding(img, pad_l, pad_t, pad_r):
    height, width = img.shape
    #Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype=np.int)
    img = np.concatenate((pad_left, img), axis=1)

    #Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis=0)

    #Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis=1)

    return img
    
def create_boundary(filename):
    original_img = Image.open(filename)
    original_img = ImageOps.grayscale(original_img)

    # convert the image to a 1:3 downsized array
    flipped_img = original_img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
    flipped_img_resized = flipped_img.resize((300, 100), Image.BICUBIC)
    arr_image = np.array(flipped_img_resized)

    # process the image to the expected format for the model 0 for flow region and 1 for boundary
    arr_image = arr_image/255
    arr_image_inv = arr_image == 0
    image_for_processing = arr_image_inv.astype(float)
    cropped_image = image_for_processing[:, 0:210]
    padded_image = add_padding(cropped_image, 23, 28, 23)
    
    original_img.close()
    flipped_img.close()
    flipped_img_resized.close()
    
    return padded_image
