#!/bin/bash
worker_hosts=host165
server_hosts=host165-s
interface=ens3
ip=$(ifconfig $interface | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1')
port=1234

clush --hostfile $worker_hosts "pkill python3; pkill bpslaunch"

DTYPE=float16
MODEL=bert_12_768_12

BS=256
ACC=1
LR=0.0001
WARMUP_RATIO=0.01
CONST_RATIO=0
NUMSTEPS=900000
OPTIMIZER=bertadam

MAX_SEQ_LENGTH=128
MAX_PREDICTIONS_PER_SEQ=20
SHORT_SEQ_PROB=0.1

LOGINTERVAL=10
CKPTDIR=$HOME/checkpoints/gluon-nlp-1/ckpt_stage1_ds_neslamb_256_bps_sz
CKPTINTERVAL=100000

DATA_HOME=$HOME/datasets/bert/pretrain/book-wiki-split-2k-v3
DATA=$DATA_HOME/*.train
DATAEVAL=$DATA_HOME/*.dev

mkdir -p $CKPTDIR

python3 ~/repos/byteps/launcher/dist_launcher.py \
  -WH $worker_hosts \
  -SH $server_hosts \
  --scheduler-ip $ip \
  --scheduler-port $port \
  --interface $interface \
  -i ~/yuchen.pem \
  --username ubuntu \
  --env NCCL_SOCKET_IFNAME:$interface \
  --env NCCL_MIN_NRINGS:1 \
  --env NCCL_DEBUG:VERSION \
  --env MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD:99999 \
  --env MXNET_SAFE_ACCUMULATION:1 \
  --env NCCL_TREE_THRESHOLD:4294967296 \
  --env OMP_WAIT_POLICY:PASSIVE \
  --env OMP_NUM_THREADS:4 \
  --env BYTEPS_THREADPOOL_SIZE:16 \
  --env BYTEPS_MIN_COMPRESS_BYTES:1024000 \
  --env BYTEPS_NUMA_ON:1 \
  --env NVIDIA_VISIBLE_DEVICES:0,1,2,3,4,5,6,7 \
  --env BYTEPS_SERVER_ENGINE_THREAD:4 \
  --env BYTEPS_PARTITION_BYTES:1024000 \
  --env BYTEPS_LOG_LEVEL:INFO \
  --env BYTEPS_FORCE_DISTRIBUTED:1 \
  "source ~/.profile;bash -c \"bpslaunch python3 ~/repos/gluon-nlp-1/scripts/bert/run_pretraining.py \
  --data=$DATA \
  --data_eval=$DATAEVAL \
  --optimizer $OPTIMIZER \
  --warmup_ratio $WARMUP_RATIO \
  --const_ratio $CONST_RATIO \
  --num_steps $NUMSTEPS \
  --ckpt_interval $CKPTINTERVAL \
  --dtype $DTYPE \
  --ckpt_dir $CKPTDIR \
  --lr $LR \
  --total_batch_size $BS \
  --total_batch_size_eval $BS \
  --accumulate $ACC \
  --model $MODEL \
  --max_seq_length $MAX_SEQ_LENGTH \
  --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
  --num_dataset_workers 2 \
  --num_batch_workers 1 \
  --circle_length 2 \
  --repeat 8 \
  --dataset_cached \
  --num_max_dataset_cached 4 \
  --short_seq_prob $SHORT_SEQ_PROB \
  --compressor onebit \
  --onebit-scaling \
  --comm_backend byteps --log_interval $LOGINTERVAL --raw\""
