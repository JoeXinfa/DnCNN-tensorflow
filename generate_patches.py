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
parser.add_argument('--save_dir', dest='save_dir', default='data',
                    help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=40,
                    help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=10,
                    help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--work_dir', dest='work_dir', default='.',
                    help='work directory')
# check output arguments
parser.add_argument('--from_file', dest='from_file',
                    default="data/img_clean_pats.npy",
                    help='get pic from file')
parser.add_argument('--num_pic', dest='num_pic', type=int, default=10,
                    help='number of pic to pick')
args = parser.parse_args()


def generate_patches():
    global DATA_AUG_TIMES
    count = 0
    src_dir = os.path.join(args.work_dir, args.src_dir)
    filepaths = glob(src_dir + '/*.png')
    num_pic = args.num_pic
    filepaths = filepaths[:num_pic]
    print("number of training data %d" % len(filepaths))

    scales = [1, 0.9, 0.8, 0.7]

    # calculate the number of patches
    for i in range(len(filepaths)):
        img = Image.open(filepaths[i]).convert('L')  # convert RGB to gray
        for s in range(len(scales)):
            newsize = (int(img.size[0] * scales[s]),
                       int(img.size[1] * scales[s]))
            # do not change the original img
            img_s = img.resize(newsize, resample=Image.BICUBIC)
            im_h, im_w = img_s.size
            x1, x2 = 0 + args.step, im_h - args.pat_size
            y1, y2 = 0 + args.step, im_w - args.pat_size
            for x in range(x1, x2, args.stride):
                for y in range(y1, y2, args.stride):
                    count += 1
    origin_patch_num = count * DATA_AUG_TIMES

    if origin_patch_num % args.bat_size != 0:
        numPatches = (origin_patch_num / args.bat_size + 1) * args.bat_size
    else:
        numPatches = origin_patch_num
    numPatches = int(numPatches)
    numBatches = int(numPatches / args.bat_size)
    print("patch size = %d, total patches = %d" % (args.pat_size, numPatches))
    print("batch size = %d, total batches = %d" % (args.bat_size, numBatches))

    # data matrix 4-D
    inputs = np.zeros((numPatches, args.pat_size, args.pat_size, 1),
                      dtype="uint8")

    count = 0
    # generate patches
    for i in range(len(filepaths)):
        img = Image.open(filepaths[i]).convert('L')
        for s in range(len(scales)):
            newsize = (int(img.size[0] * scales[s]),
                       int(img.size[1] * scales[s]))
            img_s = img.resize(newsize, resample=Image.BICUBIC)
            img_s = np.reshape(np.array(img_s, dtype="uint8"),
                               (img_s.size[0], img_s.size[1], 1))
            # extend one dimension

            for j in range(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                x1, x2 = 0 + args.step, im_h - args.pat_size
                y1, y2 = 0 + args.step, im_w - args.pat_size
                for x in range(x1, x2, args.stride):
                    for y in range(y1, y2, args.stride):
                        inputs[count, :, :, :] = data_augmentation(
                            img_s[x:x+args.pat_size, y:y+args.pat_size, :],
                            random.randint(0, 7))
                        count += 1
    # pad the batch
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

    save_dir = os.path.join(args.work_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(os.path.join(save_dir, "img_clean_pats"), inputs)
    print("size of inputs tensor = " + str(inputs.shape))


if __name__ == '__main__':
    generate_patches()
