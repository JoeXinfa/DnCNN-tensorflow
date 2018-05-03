# -*- coding: utf-8 -*-

""" Distributed TensorFlow """

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from dncnn.model import dncnn
from dncnn.utils import tf_psnr

FLAGS = None

def main(_):

    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    task_index = FLAGS.task_index
    job_name = FLAGS.job_name

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker":worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=job_name,
                             task_index=task_index)

    if job_name == "ps":
        #server.join()

        # create a shared queue on the parameter server
        # which is visible on /job:ps/task:%d
        with tf.device('/job:ps/task:%d' % task_index):
            queue = tf.FIFOQueue(cluster.num_tasks('worker'), tf.int32,
                                 shared_name='done_queue%d' % task_index)

        # wait for the queue to be filled
        with tf.Session(server.target) as sess:
            for i in range(cluster.num_tasks('worker')):
                sess.run(queue.dequeue())
                print('ps:%d received "done" from worker:%d' % (task_index, i))
            print('ps:%d quitting' % task_index)

    elif job_name == "worker":
        train(server, cluster)
    else:
        raise ValueError("Unknow job name.")

def train(server, cluster):
    print("Training model...")

    # config
    max_step = 1000000
    task_count = cluster.num_tasks('worker')
    task_index = FLAGS.task_index
    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    epoch_count = FLAGS.epoch_count
    input_c_dim = 1
    is_training = True
    noisy_file = os.path.join(FLAGS.work_dir, FLAGS.train_noisy)
    clean_file = os.path.join(FLAGS.work_dir, FLAGS.train_clean)
    ckpt_dir = os.path.join(FLAGS.work_dir, FLAGS.ckpt_dir)
    #sample_dir = os.path.join(FLAGS.work_dir, FLAGS.sample_dir)
    log_dir = os.path.join(FLAGS.work_dir, FLAGS.log_dir)

    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):

        noisy_data = np.load(noisy_file)
        clean_data = np.load(clean_file)
        # normalize to 0-1
        noisy_data = noisy_data.astype(np.float32) / 255.0
        clean_data = clean_data.astype(np.float32) / 255.0

        # Build model
        Y_ = tf.placeholder(tf.float32,
            shape=[None, None, None, input_c_dim], name='clean_image')
        X  = tf.placeholder(tf.float32,
            shape=[None, None, None, input_c_dim], name='noisy_image')
        is_training = tf.placeholder(tf.bool, name='is_training')
        Y = dncnn(X, is_training=is_training)
        # R = X - Y # residual = input - output
        loss = (1.0 / batch_size) * tf.nn.l2_loss(Y_ - Y)
        eva_psnr = tf_psnr(Y, Y_)
        lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("lr", lr)
        tf.summary.scalar("eva_psnr", eva_psnr)

        # merge all summaries into a single "operation"
        # which we can execute in a session
        summary_op = tf.summary.merge_all()
        #init_op = tf.global_variables_initializer()
        #no_op = tf.no_op()
        print("Variables initialized ...")

    batch_count = int(noisy_data.shape[0] / batch_size)
    print("Batch count %d, Task count %d" % (batch_count, task_count))

    is_chief = (task_index == 0)
    if is_chief:
        print("Worker %d: Initializing session..." % task_index)
    else:
        print("Worker %d: Waiting for session to be initialized..." % task_index)

    done_ops = []
    # create a shared queue on the worker which is visible on /job:ps/task:%d
    for i in range(cluster.num_tasks('ps')):
        with tf.device('/job:ps/task:%d' % i):
            done_queue = tf.FIFOQueue(cluster.num_tasks('worker'), tf.int32,
                                      shared_name='done_queue' + str(i))
            done_ops.append(done_queue.enqueue(task_index))

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [tf.train.StopAtStepHook(last_step=max_step),
             tf.train.FinalOpsHook([done_ops])]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing
    # when done or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(task_index == 0),
                                           checkpoint_dir=ckpt_dir,
                                           hooks=hooks) as mon_sess:

        # create log writer object (this will log on every machine)
        print("Save tensorboard files into: {}".format(log_dir))
        writer = tf.summary.FileWriter(log_dir, mon_sess.graph)
        # writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        while not mon_sess.should_stop():
            start_time = time.time()
            for epoch in range(epoch_count):
                # Separate data batches to each worker
                for batch_index in range(0, batch_count, task_count):
                    current_batch = batch_index + task_index
                    if current_batch >= batch_count:
                        current_batch -= batch_count # wrap around
                    index1 = batch_size * current_batch
                    index2 = batch_size + index1
                    noisy_batch = noisy_data[index1:index2, :, :, :]
                    clean_batch = clean_data[index1:index2, :, :, :]

                    _, loss_value, summary, step = mon_sess.run(
                        [train_op, loss, summary_op, global_step],
                        feed_dict={X: noisy_batch, Y_: clean_batch,
                                   lr: learning_rate, is_training: True})
                    writer.add_summary(summary, step)

                    elapsed_time = int((time.time() - start_time) / 60)
                    start_time = time.time()
                    print("Step: %4d," % (step+1),
                          "Epoch: %4d," % (epoch+1),
                          "Task: %4d of %4d," % (task_index+1, task_count),
                          "Batch: %4d of %4d," % (current_batch+1, batch_count),
                          "Time (min): %d," % elapsed_time,
                          "Loss: %.6f," % loss_value)

        #mon_sess.run([no_op]) # How is no_op related to done_ops?
        for op in done_ops:
            mon_sess.run(op)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--ps_hosts", type=str, default="",
        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--worker_hosts", type=str, default="",
        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--job_name", type=str, default="",
        help="One of 'ps', 'worker'")
    parser.add_argument("--task_index", type=int, default=0,
        help="Index of task within the job")

    parser.add_argument('--epoch_count', dest='epoch_count', type=int,
                        default=50, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=128, help='# images in batch')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        default=0.001, help='initial learning rate for adam')
    parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1,
                        help='gpu flag, 1 for GPU and 0 for CPU')
    parser.add_argument('--phase', dest='phase', default='train',
                        help='train or test')
    parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='checkpoint',
                        help='models are saved here')
    parser.add_argument('--sample_dir', dest='sample_dir', default='sample',
                        help='sample are saved here')
    parser.add_argument('--log_dir', dest='log_dir', default='log',
                        help='logs are saved here')
    parser.add_argument('--test_dir', dest='test_dir', default='test',
                        help='test sample are saved here')
    parser.add_argument('--train_clean', dest='train_clean',
                        default='patches_clean.npy',
                        help='clean dataset for training')
    parser.add_argument('--train_noisy', dest='train_noisy',
                        default='patches_noisy.npy',
                        help='noisy dataset for training')
    parser.add_argument('--eval_set', dest='eval_set', default='Set12',
                        help='dataset for eval in training')
    parser.add_argument('--test_set', dest='test_set', default='Set12',
                        help='dataset for testing')
    parser.add_argument('--work_dir', dest='work_dir', default='.',
                        help='work directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
