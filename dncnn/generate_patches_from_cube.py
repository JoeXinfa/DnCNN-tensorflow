# -*- coding: utf-8 -*-

"""
Generate patches from 3D data cube.
from dncnn.generate_patches_from_cube import generate_patches
"""

import random
from PIL import Image
import numpy as np
from dncnn.utils import data_augmentation

# the pixel value range is '0-255'(uint8 ) of training data

# macro
# transform a sample to a different sample for DATA_AUG_TIMES times
DATA_AUG_TIMES = 1

def generate_patches(cube, propName=None, step=0, patch_size=40, stride=10,
                     batch_size=128, save_file=None):

    NIL = cube.dictVidx['IL_AMNT']
    print("number of 2D training data %d" % NIL)

    #scales = [1, 0.9, 0.8, 0.7]
    scales = [1]

    if propName is None:
        propName = cube.currentProperty
    array3d = cube.dictProp[propName]['array3d']

    count = 0
    # calculate the number of patches
    # for i in range(NIL):
    for i in [14]:
        # from open PNG file
        #img = Image.open(filepaths[i]).convert('L')  # convert RGB to gray
        #array2d shape (NXL,NDP), transpose so NDP is row and NXL is column.
        array2d = array3d[i].T
        img = array2image(array2d)
        # save a png file for QC
        if i == 14:
            fn = '/home/zhuu/temp.png'
            img.save(fn, 'png')
            #exit()

        for s in range(len(scales)):
            newsize = (int(img.size[0] * scales[s]),
                       int(img.size[1] * scales[s]))
            # do not change the original img
            img_s = img.resize(newsize, resample=Image.BICUBIC)
            im_h, im_w = img_s.size
            x1, x2 = 0 + step, im_h - patch_size
            y1, y2 = 0 + step, im_w - patch_size
            for x in range(x1, x2, stride):
                for y in range(y1, y2, stride):
                    count += 1

        # if i == 14:
        #     fn = '/home/zhuu/temp2.png'
        #     img_s.save(fn, 'png')

    patch_count_raw = count * DATA_AUG_TIMES
    print("count, DATA_AUG_TIMES =", count, DATA_AUG_TIMES)
    print("patch_count_raw =", patch_count_raw)

    if patch_count_raw % batch_size != 0:
        patch_count = (int(patch_count_raw / batch_size) + 1) * batch_size
    else:
        patch_count = patch_count_raw
    patch_count = int(patch_count)
    batch_count = int(patch_count / batch_size)
    print("patch size = %d, total patches = %d" % (patch_size, patch_count))
    print("batch size = %d, total batches = %d" % (batch_size, batch_count))

    # data matrix 4-D
    inputs = np.zeros((patch_count, patch_size, patch_size, 1), dtype="uint8")

    count = 0
    # generate patches
    # for i in range(NIL):
    for i in [14]:
        #img = Image.open(filepaths[i]).convert('L')
        array2d = array3d[i].T
        img = array2image(array2d)
        for s in range(len(scales)):
            newsize = (int(img.size[0] * scales[s]),
                       int(img.size[1] * scales[s]))
            img_s = img.resize(newsize, resample=Image.BICUBIC)
            img_s = np.reshape(np.array(img_s, dtype="uint8"),
                               (img_s.size[0], img_s.size[1], 1))
            # extend one dimension

            for j in range(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                x1, x2 = 0 + step, im_h - patch_size
                y1, y2 = 0 + step, im_w - patch_size
                for x in range(x1, x2, stride):
                    for y in range(y1, y2, stride):
                        inputs[count, :, :, :] = data_augmentation(
                            img_s[x:x+patch_size, y:y+patch_size, :], 0)
                            # random.randint(0, 7))
                        count += 1
    # pad the batch
    if count < patch_count:
        print("Padding the batch")
        to_pad = patch_count - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

    if save_file is None:
        return inputs
    else:
        np.save(save_file, inputs)
    print("size of inputs tensor = " + str(inputs.shape))

def array2image(array2d, clipMin=None, clipMax=None):
    if clipMin is None:
        clipMin = np.min(array2d)
    if clipMax is None:
        clipMax = np.max(array2d)
    arrayClip = np.copy(array2d)
    np.clip(arrayClip, clipMin, clipMax)
    # scale value to 0-255
    arrayScale = np.uint8((arrayClip - clipMin) * 255 / (clipMax - clipMin))
    img = Image.fromarray(arrayScale).convert('L')
    return img
