from mpi4py import MPI
import socket
import os
import argparse

FLAGS = None
PYTHON = "/data/data323/devl/zhuu/bin/python"

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Testing
    # hostname = socket.gethostname()
    # print("rank, hostname =", rank, hostname)

    ps_hosts_list = FLAGS.ps_hosts.split(',')
    worker_hosts_list = FLAGS.worker_hosts.split(',')
    num_ps_hosts = len(ps_hosts_list)
    num_worker_hosts = len(worker_hosts_list)
    num_hosts = num_ps_hosts + num_worker_hosts

    if rank == 0:
        print("ps_hosts_list =", ps_hosts_list)
        print("worker_hosts_list =", worker_hosts_list)

    for rank_rotate in range(num_hosts):
        if rank == rank_rotate:
            print("I am rank " + str(rank_rotate) + "...")
            hostname = socket.gethostname()
            print("My hostname is: " + hostname)
            for ps_hosts_rotate in range(num_ps_hosts):
                if hostname in ps_hosts_list[ps_hosts_rotate].split(':')[0]:
                    print("My job ID is: ps" + str(ps_hosts_rotate))
                    os.system(PYTHON + " -u " + FLAGS.script +
                              " --ps_hosts " + FLAGS.ps_hosts +
                              " --worker_hosts " + FLAGS.worker_hosts +
                              " --job_name ps" +
                              " --task_index " + str(ps_hosts_rotate) +
                              " --epoch_count " + str(FLAGS.epoch_count) +
                              " --batch_size " + str(FLAGS.batch_size) +
                              " --learning_rate " + str(FLAGS.learning_rate) +
                              " --use_gpu " + str(FLAGS.use_gpu) +
                              " --phase " + FLAGS.phase +
                              " --checkpoint_dir " + FLAGS.ckpt_dir +
                              " --sample_dir " + FLAGS.sample_dir +
                              " --log_dir " + FLAGS.log_dir +
                              " --test_dir " + FLAGS.test_dir +
                              " --train_clean " + FLAGS.train_clean +
                              " --train_noisy " + FLAGS.train_noisy +
                              " --eval_set " + FLAGS.eval_set +
                              " --test_set " + FLAGS.test_set +
                              " --work_dir " + FLAGS.work_dir)
            for worker_hosts_rotate in range(num_worker_hosts):
                if hostname in worker_hosts_list[worker_hosts_rotate].split(':')[0]:
                    print("My job ID is: worker" + str(worker_hosts_rotate))
                    os.system(PYTHON + " -u " + FLAGS.script +
                              " --ps_hosts " + FLAGS.ps_hosts +
                              " --worker_hosts " + FLAGS.worker_hosts +
                              " --job_name worker" +
                              " --task_index " + str(worker_hosts_rotate) +
                              " --epoch_count " + str(FLAGS.epoch_count) +
                              " --batch_size " + str(FLAGS.batch_size) +
                              " --learning_rate " + str(FLAGS.learning_rate) +
                              " --use_gpu " + str(FLAGS.use_gpu) +
                              " --phase " + FLAGS.phase +
                              " --checkpoint_dir " + FLAGS.ckpt_dir +
                              " --sample_dir " + FLAGS.sample_dir +
                              " --log_dir " + FLAGS.log_dir +
                              " --test_dir " + FLAGS.test_dir +
                              " --train_clean " + FLAGS.train_clean +
                              " --train_noisy " + FLAGS.train_noisy +
                              " --eval_set " + FLAGS.eval_set +
                              " --test_set " + FLAGS.test_set +
                              " --work_dir " + FLAGS.work_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--ps_hosts", type=str, default="",
        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--worker_hosts", type=str, default="",
        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--script", type=str, default="",
        help="The .py file you want to execute")

    # options for the working script
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
    parser.add_argument('--log_dir', dest='log_dir', default='logs',
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
    main()
