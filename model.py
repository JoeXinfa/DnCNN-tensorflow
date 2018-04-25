import os
import time

import tensorflow as tf
import numpy as np

from utils import tf_psnr, cal_psnr, save_images, load_images

def dncnn(input, is_training=True, output_channels=1):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same',
                                  activation=tf.nn.relu)
    for layers in range(2, 16 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same',
                                      name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output,
                                training=is_training))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input - output

"""
When call dncnn twice...
ValueError: Variable block1/conv2d/kernel already exists, disallowed.
Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
Originally defined at:
        output = tf.layers.conv2d(input, 64, 3, padding='same',
                                  activation=tf.nn.relu)
"""

#class denoiser(object): # Python2 style
class denoiser: # Python3 style
    def __init__(self, sess, input_c_dim=1, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        # build model
        self.Y_ = tf.placeholder(tf.float32,
            [None, None, None, self.input_c_dim], name='clean_image')
        self.X  = tf.placeholder(tf.float32,
            [None, None, None, self.input_c_dim], name='noisy_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        #self.X = self.Y_ + tf.random_normal(shape=tf.shape(self.Y_),
        #    stddev=sigma / 255.0)  # noisy images
        self.Y = dncnn(self.X, is_training=self.is_training)
        self.R = self.X - self.Y # residual = input - output
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, evaln_data, evalc_data, sample_dir,
                 summary_merged, summary_writer):
        """
        -i- evaln_data : list, of 4D array of different size.
            Each array is a noisy image for evaluation, value range 0-255.
        -i- evalc_data : list, of 4D array of different size.
            Each array is a clean image for evaluation, value range 0-255.
        """
        # assert eval_data value range is 0-255
        print("[*] Evaluating...")
        psnr_sum = 0
        for idx in range(len(evaln_data)):
            noisy_image = evaln_data[idx].astype(np.float32) / 255.0
            clean_image = evalc_data[idx].astype(np.float32) / 255.0
            output_image, psnr_summary = self.sess.run(
                [self.Y, summary_merged],
                feed_dict={self.X: noisy_image, self.Y_: clean_image,
                           self.is_training: False})
            summary_writer.add_summary(psnr_summary, iter_num)

            groundtruth = np.clip(evalc_data[idx], 0, 255).astype('uint8')
            noisy_img = np.clip(evaln_data[idx], 0, 255).astype('uint8')
            output_img = np.clip(255 * output_image, 0, 255).astype('uint8')

            # calculate PSNR
            psnr = cal_psnr(groundtruth, output_img)
            print("img%d PSNR: %.2f" % (idx + 1, psnr))
            psnr_sum += psnr
            filename = 'test%d_%d.png' % (idx + 1, iter_num)
            filename = os.path.join(sample_dir, filename)
            save_images(filename, groundtruth, noisy_img, output_img)
        avg_psnr = psnr_sum / len(evaln_data)

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

    def denoise(self, data):
        output_clean_image, noisy_image, psnr = self.sess.run(
                [self.Y, self.X, self.eva_psnr],
                feed_dict={self.Y_: data, self.is_training: False})
        return output_clean_image, noisy_image, psnr

    def train(self, noisy_data, clean_data, evaln_data, evalc_data,
              ckpt_dir='./checkpoint', sample_dir='./sample',
              log_dir='./logs', epoch=50, batch_size=128, lr=0.001,
              eval_every_epoch=2):
        """
        -i- noisy_data : array, numpy 4D (numPatches, patchSize, patchSize,
            colorDimension)
        -i- clean_data : array, numpy 4D (numPatches, patchSize, patchSize,
            colorDimension)
        -i- evaln_data : list, of 4D array of different size.
            Each array is a noisy image for evaluation, value range 0-255.
        -i- evalc_data : list, of 4D array of different size.
            Each array is a clean image for evaluation, value range 0-255.
        """
        # assert data range is between 0 and 1
        numBatch = int(noisy_data.shape[0] / batch_size)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % \
              (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, evaln_data, evalc_data, sample_dir,
                      summary_psnr, writer)
        for epoch in range(start_epoch, epoch):
            #np.random.shuffle(noisy_data)
            for batch_id in range(start_step, numBatch):
                index1 = batch_id * batch_size
                index2 = (batch_id + 1) * batch_size
                noisy_batch = noisy_data[index1:index2, :, :, :]
                clean_batch = clean_data[index1:index2, :, :, :]
                # normalize the data to 0-1, use less memory
                # noisy_batch = noisy_batch.astype(np.float32) / 255.0
                _, loss, summary = self.sess.run(
                    [self.train_op, self.loss, merged],
                    feed_dict={self.X: noisy_batch, self.Y_: clean_batch,
                               self.lr: lr[epoch], self.is_training: True})
                print("Epoch: %2d, batch: %4d/%4d, time (min): %d, loss: %.6f"
                      % (epoch + 1, batch_id + 1, numBatch,
                         (time.time() - start_time) / 60, loss))
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, evaln_data, evalc_data, sample_dir,
                              summary_psnr, writer)
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")

    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_files, ckpt_dir='./checkpoint', save_dir='./test'):
        """
        -i- test_files : list, of string for the filename of noisy image.
        -i- ckpt_dir : string, path of the folder containing training info.
        -i- save_dir : string, path of the folder to save the result.
        Input noisy image, output denoised image.
        """
        # init variables
        #tf.initialize_all_variables().run() # deprecated
        init = tf.global_variables_initializer()
        self.sess.run(init)
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")
        psnr_sum = 0
        for idx in range(len(test_files)):
            noisy_image = load_images(test_files[idx])
            noisy_image = noisy_image.astype(np.float32) / 255.0
            output_image, diff_image = self.sess.run([self.Y, self.R],
                feed_dict={self.X: noisy_image, self.is_training: False})
            noisy_img = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            output_img = np.clip(255 * output_image, 0, 255).astype('uint8')
            diff_img = np.clip(255 * diff_image, 0, 255).astype('uint8')
            # RGB=(0,0,0) for black, (255,255,255) for white
            diff_img = 255 - diff_img # flip to use white as background
            # calculate PSNR
            psnr = cal_psnr(noisy_img, output_img)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            filename = os.path.join(save_dir, 'test%d.png' % (idx+1))
            save_images(filename, noisy_img, output_img, diff_img)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
