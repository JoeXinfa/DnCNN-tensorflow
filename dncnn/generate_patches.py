import os
import argparse
from glob import glob
import random
from PIL import Image
import numpy as np
from utils import data_augmentation

# the pixel value range is '0-255'(uint8 ) of training data

# macro
# transform a sample to a different sample for DATA_AUG_TIMES times
DATA_AUG_TIMES = 1

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='data/Train400',
                    help='dir of data')
parser.add_argument('--save_file', dest='save_file', default='data/pats',
                    help='filename of patches')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=40,
                    help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=10,
                    help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--work_dir', dest='work_dir', default='.',
                    help='work directory')
# check output arguments
parser.add_argument('--from_file', dest='from_file',
                    default="data/img_clean_pats.npy",
                    help='get pic from file')
parser.add_argument('--num_pic', dest='num_pic', type=int, default=1,
                    help='number of pic to pick')
args = parser.parse_args()


def generate_patches():
    batch_size = args.batch_size
    patch_size = args.patch_size

    global DATA_AUG_TIMES
    count = 0
    src_dir = os.path.join(args.work_dir, args.src_dir)
    filepaths = glob(src_dir + '/*.png')
    num_pic = args.num_pic
    filepaths = filepaths[:num_pic]
    print("Number of training data %d" % len(filepaths))
    print("Number of training data used %d" % num_pic)

    #scales = [1, 0.9, 0.8, 0.7]
    scales = [1]

    # calculate the number of patches
    for i in range(len(filepaths)):
        img = Image.open(filepaths[i]).convert('L')  # convert RGB to gray
        for s in range(len(scales)):
            newsize = (int(img.size[0] * scales[s]),
                       int(img.size[1] * scales[s]))
            # do not change the original img
            img_s = img.resize(newsize, resample=Image.BICUBIC)
            im_h, im_w = img_s.size
            x1, x2 = 0 + args.step, im_h - patch_size
            y1, y2 = 0 + args.step, im_w - patch_size
            for x in range(x1, x2, args.stride):
                for y in range(y1, y2, args.stride):
                    count += 1
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
    for i in range(len(filepaths)):
        img = Image.open(filepaths[i]).convert('L')
        for s in range(len(scales)):
            newsize = (int(img.size[0] * scales[s]),
                       int(img.size[1] * scales[s]))
            img_s = img.resize(newsize, resample=Image.BICUBIC)
            # extend one dimension
            img_s = np.reshape(np.array(img_s, dtype="uint8"),
                               (img_s.size[0], img_s.size[1], 1))

            for j in range(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                x1, x2 = 0 + args.step, im_h - patch_size
                y1, y2 = 0 + args.step, im_w - patch_size
                for x in range(x1, x2, args.stride):
                    for y in range(y1, y2, args.stride):
                        inputs[count, :, :, :] = data_augmentation(
                            img_s[x:x+patch_size, y:y+patch_size, :], 0)
                            #random.randint(0, 7))
                        count += 1
    # pad the batch
    if count < patch_count:
        to_pad = patch_count - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

    save_file = os.path.join(args.work_dir, args.save_file)
    np.save(save_file, inputs)
    print("size of inputs tensor = " + str(inputs.shape))


if __name__ == '__main__':
    generate_patches()
