import argparse
import os
from glob import glob

import tensorflow as tf
import numpy as np

from model import denoiser
from utils import load_data, load_images

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50,
                    help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                    help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                    help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1,
                    help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--sigma', dest='sigma', type=int, default=25,
                    help='noise level')
parser.add_argument('--phase', dest='phase', default='train',
                    help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='checkpoint',
                    help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='sample',
                    help='sample are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='logs',
                    help='logs are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='test',
                    help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='Set12',
                    help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='Set12',
                    help='dataset for testing')
parser.add_argument('--work_dir', dest='work_dir', default='.',
                    help='work directory')
args = parser.parse_args()


def denoiser_train(denoiser, lr):
    data_file = os.path.join(args.work_dir, 'data/img_clean_pats.npy')
    eval_files = 'data/test/{}/*.png'.format(args.eval_set)
    eval_files = os.path.join(args.work_dir, eval_files)
    ckpt_dir = os.path.join(args.work_dir, args.ckpt_dir)
    sample_dir = os.path.join(args.work_dir, args.sample_dir)
    log_dir = os.path.join(args.work_dir, args.log_dir)
    batch_size = args.batch_size
    epoch = args.epoch
    with load_data(filepath=data_file) as data:
        # If there is a small memory, please comment this line and uncomment
        # the corresponding line in model.py
        data = data.astype(np.float32) / 255.0  # normalize the data to 0-1
        eval_files = glob(eval_files)
        # list of array of different size, 4-D, pixel value range is 0-255
        eval_data = load_images(eval_files)
        denoiser.train(data, eval_data, batch_size=batch_size,
                       ckpt_dir=ckpt_dir, epoch=epoch, lr=lr,
                       sample_dir=sample_dir, log_dir=log_dir)


def denoiser_test(denoiser):
    test_files = 'data/test/{}/*.png'.format(args.test_set)
    test_files = os.path.join(args.work_dir, test_files)
    ckpt_dir = os.path.join(args.work_dir, args.ckpt_dir)
    test_dir = os.path.join(args.work_dir, args.test_dir)
    test_files = glob(test_files)
    denoiser.test(test_files, ckpt_dir=ckpt_dir, save_dir=test_dir)


def main(_):
    ckpt_dir = os.path.join(args.work_dir, args.ckpt_dir)
    sample_dir = os.path.join(args.work_dir, args.sample_dir)
    test_dir = os.path.join(args.work_dir, args.test_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[30:] = lr[0] / 10.0
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            model = denoiser(sess, sigma=args.sigma)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, sigma=args.sigma)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()
