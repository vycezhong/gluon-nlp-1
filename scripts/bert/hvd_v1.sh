worker_hosts=/home/ec2-user/efs/ads/bench_v2/scripts/bert/hosts_64_v2

clush --hostfile $worker_hosts "pkill python"

DTYPE=float16
MODEL=bert_24_1024_16

BS=49152
#BS=65536
#BS=16384
ACC=1
LR=0.007
#WARMUP_RATIO=0.2843
#NUMSTEPS=7038
#LR=0.006
WARMUP_RATIO=0.568
NUMSTEPS=4977
OPTIMIZER=nlamb

MAX_SEQ_LENGTH=128
MAX_PREDICTIONS_PER_SEQ=20
SHORT_SEQ_PROB=0.1

LOGINTERVAL=10
CKPTDIR="/home/ec2-user/efs/shuai/gluon-nlp-1/ckpt_stage1_ds_lamb_64k_hvd_sz"
CKPTINTERVAL=300000000

DATA_HOME=/fsx/datasets/book-wiki-split-2k-v3
DATA=$DATA_HOME/*.train
DATAEVAL=$DATA_HOME/*.dev

#DATA_HOME=/home/ec2-user/efs/shuai/dataset/phase1
#DATA=$DATA_HOME/*.npz
#DATAEVAL=/home/ec2-user/efs/shuai/gluon-nlp-1/ckpt_stage1_lamb_64k_hvd_sz/data_eval_cache/part-000.npz

mkdir -p $CKPTDIR

mpirun --allow-run-as-root -np 512 --hostfile $worker_hosts \
            -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
            --bind-to none \
            -x NCCL_SOCKET_IFNAME=eth0 \
            -x NCCL_IB_HCA=eth0 \
            -x FI_PROVIDER="efa" -x FI_EFA_TX_MIN_CREDITS=64 \
            -x LD_LIBRARY_PATH=$HOME/aws-ofi-nccl/install/lib/:$HOME/nccl/build/lib:/usr/local/cuda-10.0/lib64:/opt/amazon/efa/lib64:$LD_LIBRARY_PATH \
            -x NCCL_MIN_NRINGS=1 \
            -x NCCL_DEBUG=VERSION \
            -x HOROVOD_HIERARCHICAL_ALLREDUCE=0 \
            -x HOROVOD_CYCLE_TIME=85 \
            -x HOROVOD_NUM_NCCL_STREAMS=2 \
            -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=99999 \
            -x MXNET_SAFE_ACCUMULATION=1 \
            -x NCCL_TREE_THRESHOLD=15360000 \
            --tag-output ./ompi_bind_DGX1.sh \
            python3 run_pretraining.py \
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
            --num_dataset_workers 2 \
            --num_batch_workers 1 \
            --circle_length 2 \
            --repeat 8092 \
            --dataset_cached \
            --num_max_dataset_cached 4 \
            --short_seq_prob $SHORT_SEQ_PROB \
            --comm_backend horovod --log_interval $LOGINTERVAL --raw
