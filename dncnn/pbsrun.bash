#!/bin/bash

testdir=`pwd`
job="$testdir/mpirun.bash"
outdir="$testdir"
qsub=/usr/bin/qsub

#      -l select=10:mpiprocs=1:ncpus=16:node_class=n16.128:accel=none:network=FEX \
#      -l jg_n16_128_none_FEX_gsu_a=160 \

#      -l select=5:mpiprocs=1:ncpus=24:node_class=n24.256:accel=none:network=FEX \
#      -l jg_n24_256_none_FEX_rd_a=120 \

#      -l select=3:mpiprocs=1:ncpus=24:node_class=n24.256:accel=none:network=FEX \
#      -l jg_n24_256_none_FEX_dwep_a=72 \

$qsub -q parallel@sasl0002 \
      -l select=3:mpiprocs=1:ncpus=16:node_class=n16.128:accel=none:network=FEX \
      -l jg_n16_128_none_FEX_gsu_a=48 \
      -N zhuu_test -r n \
      -j oe -o "$outdir/job.output.zhuu_test" \
      -W umask=002 \
      $job
