import argparse
import os
from glob import glob

import tensorflow as tf
import numpy as np

from model import denoiser
from utils import load_images

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
                    help='train, test or prod')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='checkpoint',
                    help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='sample',
                    help='sample are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='logs',
                    help='logs are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='test',
                    help='test sample are saved here')
parser.add_argument('--prod_dir', dest='prod_dir', default='prod',
                    help='production results are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='Set12',
                    help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='Set12',
                    help='dataset for testing')
parser.add_argument('--prod_set', dest='prod_set', default='Joe1',
                    help='dataset for production')
parser.add_argument('--work_dir', dest='work_dir', default='.',
                    help='work directory')
args = parser.parse_args()


def denoiser_train(denoiser, lr):
    noisy_file = os.path.join(args.work_dir, 'data/img_noisy_pats.npy')
    clean_file = os.path.join(args.work_dir, 'data/img_clean_pats.npy')
    evalc_files = 'data/test/{}/clean*.png'.format(args.eval_set)
    evalc_files = os.path.join(args.work_dir, evalc_files)
    evaln_files = 'data/test/{}/noisy*.png'.format(args.eval_set)
    evaln_files = os.path.join(args.work_dir, evaln_files)
    ckpt_dir = os.path.join(args.work_dir, args.ckpt_dir)
    sample_dir = os.path.join(args.work_dir, args.sample_dir)
    log_dir = os.path.join(args.work_dir, args.log_dir)
    batch_size = args.batch_size
    epoch = args.epoch

    noisy_data = np.load(noisy_file)
    clean_data = np.load(clean_file)

    # If there is a small memory, please comment this line and uncomment
    # the corresponding line in model.py
    noisy_data = noisy_data.astype(np.float32) / 255.0  # normalize to 0-1
    clean_data = clean_data.astype(np.float32) / 255.0  # normalize to 0-1
    evalc_files = glob(evalc_files)
    evaln_files = glob(evaln_files)
    # list of array of different size, 4-D, pixel value range is 0-255
    evalc_data = load_images(evalc_files)
    evaln_data = load_images(evaln_files)
    denoiser.train(noisy_data, clean_data, evaln_data, evalc_data,
                   batch_size=batch_size, ckpt_dir=ckpt_dir, epoch=epoch,
                   lr=lr, sample_dir=sample_dir, log_dir=log_dir)


def denoiser_test(denoiser):
    test_files = 'data/test/{}/*.png'.format(args.test_set)
    test_files = os.path.join(args.work_dir, test_files)
    ckpt_dir = os.path.join(args.work_dir, args.ckpt_dir)
    test_dir = os.path.join(args.work_dir, args.test_dir)
    test_files = glob(test_files)
    denoiser.test(test_files, ckpt_dir=ckpt_dir, save_dir=test_dir)

def denoiser_prod(denoiser):
    prod_files = 'data/prod/{}/*.png'.format(args.prod_set)
    prod_files = os.path.join(args.work_dir, prod_files)
    ckpt_dir = os.path.join(args.work_dir, args.ckpt_dir)
    prod_dir = os.path.join(args.work_dir, args.prod_dir)
    prod_files = glob(prod_files)
    denoiser.prod(prod_files, ckpt_dir=ckpt_dir, save_dir=prod_dir)

def main(_):
    ckpt_dir = os.path.join(args.work_dir, args.ckpt_dir)
    sample_dir = os.path.join(args.work_dir, args.sample_dir)
    test_dir = os.path.join(args.work_dir, args.test_dir)
    prod_dir = os.path.join(args.work_dir, args.prod_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(prod_dir):
        os.makedirs(prod_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[30:] = lr[0] / 10.0
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            model = denoiser(sess, sigma=args.sigma, phase=args.phase)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            elif args.phase == 'prod':
                denoiser_prod(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, sigma=args.sigma, phase=args.phase)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            elif args.phase == 'prod':
                denoiser_prod(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()
