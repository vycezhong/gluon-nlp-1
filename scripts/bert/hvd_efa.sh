clush --hostfile worker_8 "pkill python"

DTYPE=float16
MODEL=bert_24_1024_16

BS=32768
ACC=64
LR=0.004
WARMUP_RATIO=0.128
NUMSTEPS=1563

#BS=65536
#ACC=16
#LR=0.006
#WARMUP_RATIO=0.2843
#NUMSTEPS=7038
OPTIMIZER=lamb2

MAX_SEQ_LENGTH=512
MAX_PREDICTIONS_PER_SEQ=80
SHORT_SEQ_PROB=0.1

LOGINTERVAL=10
CKPTDIR="/home/ubuntu/efs/gluon-nlp-cus/ckpt_stage2_lamb_32k_hvd_sz"
CKPTINTERVAL=300000000

export TRUNCATE_NORM="${TRUNCATE_NORM:-1}"
export LAMB_BULK="${LAMB_BULK:-30}"
export EPS_AFTER_SQRT="${EPS_AFTER_SQRT:-1}"

DATA_HOME=/home/ubuntu/mxnet-data/bert-pretraining/datasets/book-wiki-split-2k-v3
DATA=$DATA_HOME/*.train
DATAEVAL=$DATA_HOME/*.dev

mkdir -p $CKPTDIR

mpirun --allow-run-as-root --tag-output -np 64 --hostfile worker_8 \
        -map-by ppr:4:socket -mca pml ob1 -mca btl ^openib \
        -x NCCL_SOCKET_IFNAME=^lo,docker0 -mca btl_tcp_if_exclude lo,docker0 --bind-to none \
        --mca plm_rsh_agent 'ssh -q -o StrictHostKeyChecking=no' \
        -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=1 \
        -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
        -x HOROVOD_CYCLE_TIME=1 \
        -x EPS_AFTER_SQRT=1 -x LAMB_BULK=30 \
        -x MXNET_SAFE_ACCUMULATION=1 \
        -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=120 \
        -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=120 \
        python3 -u run_pretraining.py \
            --data=$DATA \
            --data_eval=$DATAEVAL \
            --optimizer $OPTIMIZER \
            --warmup_ratio $WARMUP_RATIO \
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
            --num_dataset_workers 8 \
            --num_batch_workers 2 \
            --circle_length 4 \
            --repeat 8092 \
            --dataset_cached \
            --num_max_dataset_cached 8 \
            --start_step 7038 \
            --phase2 \
            --phase1_num_steps 7038 \
            --short_seq_prob $SHORT_SEQ_PROB \
            --comm_backend horovod --log_interval $LOGINTERVAL --raw
