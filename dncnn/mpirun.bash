#!/bin/bash

# Pause PBS, ssh to the nodes and test scripts.
#echo $PBS_NODEFILE
#cp $PBS_NODEFILE /home/zhuu/PBS_NODEFILE.txt
#sleep 3600

python=/data/data323/devl/zhuu/bin/python
mpiexec=/data/data323/devl/zhuu/bin/mpiexec
codedir=/home/zhuu/code/DnCNN-tensorflow/dncnn

export PS_HOSTS=$($python $codedir/cluster_specs.py --hosts_file=$PBS_NODEFILE --num_ps_hosts=1 | cut -f1 -d ' ')
export WORKER_HOSTS=$($python $codedir/cluster_specs.py --hosts_file=$PBS_NODEFILE --num_ps_hosts=1 | cut -f2 -d ' ')

echo "PBS_NODEFILE is" $PBS_NODEFILE
echo "PS_HOSTS is" $PS_HOSTS
echo "WORKER_HOSTS is" $WORKER_HOSTS
cp $PBS_NODEFILE /home/zhuu/PBS_NODEFILE.txt

# Below will run on every host in the PBS_NODEFILE.
$mpiexec -ppn 1 \
    $python -u $codedir/cluster_dispatch.py \
    --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS \
    --script=$codedir/cluster_train.py \
    --work_dir /cpfs/lfs02/data/zhuu/seam \
    --train_clean data/patches_clean_il1.npy \
    --train_noisy data/patches_randm_il1.npy \
    --eval_set eva1 \
    --checkpoint_dir checkpoint1 \
    --sample_dir sample1 \
    --log_dir log1 \
    --use_gpu 0 \
    --batch_size 3 \
    --epoch_count 2 |& tee /home/zhuu/pbs.log
