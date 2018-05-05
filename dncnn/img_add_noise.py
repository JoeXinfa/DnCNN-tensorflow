# -*- coding: utf-8 -*-

"""
Add random noise to image
"""

import os
from glob import glob
from PIL import Image
import numpy as np

def img_add_noise(img_dir, mu=0, sigma=25):
    #img_dir = "/cpfs/lfs02/data/zhuu/seam/data/Train400"
    filepaths = glob(img_dir + '/*.png')
    # filepaths = glob(img_dir + '/test_004.png') # test

    for i in range(len(filepaths)):
        input_fn = filepaths[i]
        pre, ext = os.path.splitext(input_fn)
        output_fn = pre + '_noisy' + ext
        print("input_fn =", input_fn)
        print("output_fn", output_fn)

        img = Image.open(input_fn).convert('L')  # convert RGB to gray
        input_array = np.array(img, dtype="uint8")

        # mu, sigma = 0, 25 # mean and standard deviation
        noise = np.random.normal(mu, sigma, input_array.shape)
        noise = noise.astype('int32')

        output_array = np.zeros(input_array.shape, dtype='int32')
        w, h = input_array.shape
        for i in range(w):
            for j in range(h):
                output_array[i,j] = input_array[i,j] + noise[i,j]
        output_array = np.clip(output_array, 0, 255)

        im = Image.fromarray(output_array.astype('uint8')).convert('L')
        im.save(output_fn, 'png')
