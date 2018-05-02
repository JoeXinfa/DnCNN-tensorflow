import tensorflow as tf
import argparse
import sys
import time
import os
import numpy as np
# from tensorflow.contrib.session_bundle import exporter

from dncnn.model import dncnn
from dncnn.utils import tf_psnr

FLAGS = None

def create_done_queue(i, worker_count):
  """Queue used to signal death for i'th ps shard. Intended to have
  all workers enqueue an item onto it to signal doneness."""
  with tf.device("/job:ps/task:%d" % (i)):
    return tf.FIFOQueue(worker_count, tf.int32,
                        shared_name="done_queue"+str(i))

def create_done_queues(ps_count, worker_count):
  return [create_done_queue(i, worker_count) for i in range(ps_count)]


def main(_):

    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')

    # ps_count = len(ps_hosts)
    worker_count = len(worker_hosts)

    cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker":worker_hosts})

    # start a server for a specific task
    server = tf.train.Server(
        cluster,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        sess = tf.Session(server.target)
        queue = create_done_queue(FLAGS.task_index, worker_count)

        # wait until all workers are done
        for i in range(worker_count):
            sess.run(queue.dequeue())
            print("ps %d received done worker %d" % (FLAGS.task_index, i))
        print("ps %d: quitting" % (FLAGS.task_index))

    elif FLAGS.job_name == "worker":
        train(server, cluster)

def train(server, cluster):
    print("Training model...")

    # config
    #max_step = 1000
    task_count = len(cluster._cluster_spec['worker'])
    worker_count = len(cluster._cluster_spec['worker'])
    ps_count = len(cluster._cluster_spec['ps'])
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
        global_step = tf.Variable(0, name="global_step", trainable=False)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("lr", lr)
        tf.summary.scalar("eva_psnr", eva_psnr)

        # merge all summaries into a single "operation"
        # which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(sharded=True)
        print("Variables initialized ...")

        enqueue_ops = []
        for q in create_done_queues(ps_count, worker_count):
            qop = q.enqueue(1)
            enqueue_ops.append(qop)

    # number of batches in one epoch
    batch_count = int(noisy_data.shape[0] / batch_size)
    print("Batch count %d, Task count %d" % (batch_count, task_count))

    if batch_count >= task_count:
        batch_stop = batch_count - task_count
        batch_drop = batch_count % task_count
        print("WARNING: the last %d batches are unused." % batch_drop)
    else:
        #batch_stop = 1 # not work with enqueue/dequeue method
        raise ValueError("Batch count is smaller than task count.")

    is_chief = (task_index == 0)
    if is_chief:
        print("Worker %d: Initializing session..." % task_index)
    else:
        print("Worker %d: Waiting for session to be initialized..." % task_index)

    sv = tf.train.Supervisor(is_chief=(task_index == 0),
        logdir=ckpt_dir, init_op=init_op, summary_op=None, saver=saver,
        global_step=global_step, save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.

    #begin_time = time.time()
    with sv.prepare_or_wait_for_session(server.target) as sess:

        print("Worker %d: Session initialization complete." % task_index)

        # Loop until the supervisor shuts down or n steps have completed.
        #step = 0
        #while not sv.should_stop() and step < max_step:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details
        # on how to perform *synchronous* training.

        # create log writer object (this will log on every machine)
        print("Save tensorboard files into: {}".format(log_dir))
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        # writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        # builder = tf.saved_model.builder.SavedModelBuilder(ckpt_dir)
        # builder.add_meta_graph_and_variables(sess,
        #     [tf.saved_model.tag_constants.SERVING],
        #     signature_def_map = {"magic_model":
        #     tf.saved_model.signature_def_utils.predict_signature_def(
        #     inputs= {"image": X}, outputs= {"prediction": Y})},
        #     clear_devices=True)
        # RuntimeError: Graph is finalized and cannot be modified.

        # perform training cycles
        start_time = time.time()
        for epoch in range(epoch_count):
            # Separate data batches to each worker
            for batch_index in range(0, batch_stop, task_count):
                current_batch = batch_index + task_index + 1
                if current_batch <= batch_count:

                    index1 = batch_size * (current_batch - 1)
                    index2 = batch_size * current_batch
                    noisy_batch = noisy_data[index1:index2, :, :, :]
                    clean_batch = clean_data[index1:index2, :, :, :]

                    _, loss_value, summary, step = sess.run(
                        [train_op, loss, summary_op, global_step],
                        feed_dict={X: noisy_batch, Y_: clean_batch,
                                   lr: learning_rate, is_training: True})
                    writer.add_summary(summary, step)

                    elapsed_time = int((time.time() - start_time) / 60)
                    start_time = time.time()
                    print("Step: %4d," % (step+1),
                          "Epoch: %4d," % (epoch+1),
                          "Task: %4d of %4d," % (task_index+1, task_count),
                          "Batch: %4d of %4d," % (current_batch, batch_count),
                          "Time (min): %d," % elapsed_time,
                          "Loss: %.6f," % loss_value)

            # if np.mod(epoch + 1, 2) == 0:
            #     saver = tf.train.Saver()
            #     model_name = 'DnCNN-tensorflow'
            #     path = os.path.join(ckpt_dir, model_name)
            #     saver.save(sess, path, global_step=step)

        # print("Test-Accuracy: %2.2f" % sess.run(accuracy,
        #     feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        # print("Total Time: %3.2fs" % float(time.time() - begin_time))
        # print("Final Cost: %.4f" % cost)

        # model_exporter = exporter.Exporter(saver)
        # model_exporter.init(tf.get_default_graph().as_graph_def(),
        #     named_graph_signatures={
        #     'inputs': exporter.generic_signature({'input': X}),
        #     'outputs': exporter.generic_signature({'output': Y})},
        #     clear_devices=True)
        # export_path = ckpt_dir
        # model_exporter.export(export_path, sess)

        # signal to ps shards that we are done
        #for q in create_done_queues(ps_count, worker_count):
        #    sess.run(q.enqueue(1))
        for op in enqueue_ops:
            sess.run(op)

    # builder.save()
    sv.stop()
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
