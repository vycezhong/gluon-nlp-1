# python -c 'import os; print(os.environ); import socket; print(socket.gethostname()); import mxnet as mx; import horovod.mxnet as hvd; print(mx); hvd.init(); print(hvd.rank())'

pkill python

	    #-x HOROVOD_TIMELINE=timeline.efa \

mpirun -np $NP --hostfile $HOST -display-allocation --allow-run-as-root \
	    -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
            --bind-to none \
	    --mca plm_rsh_agent "ssh -q -o StrictHostKeyChecking=no -p $PORT" \
            -x NCCL_SOCKET_IFNAME=eth0 \
            -x FI_PROVIDER="efa" -x FI_EFA_TX_MIN_CREDITS=64 \
            -x NCCL_IB_HCA=eth0 \
            -x NCCL_DEBUG=INFO \
	    -x NCCL_MIN_NRINGS=$NCCLMINNRINGS \
            -x HOROVOD_FUSION_THRESHOLD=268435456 \
	    -x HOROVOD_HIERARCHICAL_ALLREDUCE=$HIERARCHICAL \
            -x HOROVOD_NUM_NCCL_STREAMS=2 \
	    -x HOROVOD_CYCLE_TIME=30 \
	    -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=120 \
            -x NO_DROPOUT=$NO_DROPOUT \
            -x LD_LIBRARY_PATH=$HOME/aws-ofi-nccl/install/lib/:$HOME/nccl/build/lib:/usr/local/cuda-10.0/lib64:/opt/amazon/efa/lib64:$LD_LIBRARY_PATH \
	    -x LARGE_WINDOW=1 \
	    -x MXNET_SAFE_ACCUMULATION=1 \
            -x NCCL_TREE_THRESHOLD=0 \
            -x TRUNCATE_NORM=$TRUNCATE_NORM \
            -x LAMB_BULK=$LAMB_BULK \
            -x EPS_AFTER_SQRT=$EPS_AFTER_SQRT \
            -x NO_SHARD=$NO_SHARD \
	    --tag-output ompi_bind_DGX1.sh python3 run_pretraining.py \
	    --data="$DATA_HOME/book-corpus/book-corpus-large-split/*.train,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.train" \
	    --data_eval="$DATA_HOME/book-corpus/book-corpus-large-split/*.test,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.test" \
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
	    --num_data_workers 4 \
            --eval_interval $EVALINTERVAL \
	    --no_compute_acc --raw \
	    --comm_backend horovod --log_interval $LOGINTERVAL $OPTIONS 2>&1 | tee -a mpi.log
