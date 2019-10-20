#TRUNCATE_NORM=1 LAMB_BULK=30 EPS_AFTER_SQRT=1 NUMSTEPS=900000 DTYPE=float16 BS=256 ACC=2 MODEL=bert_24_1024_16 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.0001 LOGINTERVAL=1 CKPTDIR=ckpt_stage1_adam_1x_kv CKPTINTERVAL=300000 OPTIMIZER=bertadam WARMUP_RATIO=0.0001 bash kvstore.sh

export HOST=hosts_32
export OTHER_HOST=hosts_31
export DOCKER_IMAGE=haibinlin/worker_mxnet:c5fd6fc-1.5-cu90-79e6e8-79e6e8
export PORT=12450
export NP=256
export NCCLMINNRINGS=1
export TRUNCATE_NORM=1
export LAMB_BULK=30
export EPS_AFTER_SQRT=1
export DTYPE=float16
export MODEL=bert_24_1024_16
export LOGINTERVAL=50
export CKPTDIR=/bert/ckpt_stage1_64k_32k_e818446
export CKPTINTERVAL=300000000
export OPTIMIZER=lamb2
export COMMIT=e818446
export CLUSHUSER=ec2-user
export NO_SHARD=1
export HIERARCHICAL=1
export EVALINTERVAL=1000

bash clush-hvd.sh

export LOGINTERVAL=1
#export OPTIONS='--synthetic_data --eval_use_npz --verbose'
export OPTIONS='--synthetic_data --verbose'
export NUMSTEPS=20

BS=65536 ACC=4 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.006 WARMUP_RATIO=0.2843 bash mul-hvd.sh

#NUMSTEPS=15625 BS=32768 ACC=8 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
#NUMSTEPS=7813 BS=65536 ACC=4 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.006 WARMUP_RATIO=0.2843 bash mul-hvd.sh

bash clush-hvd.sh
export OPTIONS="--synthetic_data --verbose --phase2 --phase1_num_steps=$NUMSTEPS --start_step=$NUMSTEPS"

#NUMSTEPS=1563 BS=32768 ACC=8 MAX_SEQ_LENGTH=512 MAX_PREDICTIONS_PER_SEQ=80 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
export NUMSTEPS=10
BS=32768 ACC=8 MAX_SEQ_LENGTH=512 MAX_PREDICTIONS_PER_SEQ=80 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh

python3 finetune_squad.py --bert_model bert_24_1024_16 --pretrained_bert_parameters $CKPTDIR/0000010.params --optimizer adam --accumulate 3 --batch_size 8 --lr 3e-5 --epochs 2 --gpu 0
