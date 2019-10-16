pkill python
mpirun --allow-run-as-root --tag-output -np 256 --hostfile $HOME/hosts_np256 \
        -map-by ppr:4:socket -mca pml ob1 -mca btl ^openib  -mca btl_tcp_if_include eth0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x FI_PROVIDER="efa" -x FI_EFA_TX_MIN_CREDITS=64 \
        -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=1 \
        -x HOROVOD_FUSION_THRESHOLD=268435456 \
        -x HOROVOD_HIERARCHICAL_ALLREDUCE=0 \
        -x HOROVOD_CYCLE_TIME=30 \
        -x NCCL_TREE_THRESHOLD=0 \
        -x EPS_AFTER_SQRT=1 -x LAMB_BULK=30 \
        -x MXNET_SAFE_ACCUMULATION=1 \
        -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=120 \
        python run_pretraining.py --comm_backend horovod \
                --model='bert_24_1024_16' \
                --data='/home/ec2-user/mxnet-data/bert-pretraining/datasets/book-corpus/book-corpus-large-split/*.train,/home/ec2-user/mxnet-data/bert-pretraining/datasets/enwiki/enwiki-feb-doc-split/*.train' \
                --data_eval='/home/ec2-user/mxnet-data/bert-pretraining/datasets/book-corpus/book-corpus-large-split/*.dev,/home/ec2-user/mxnet-data/bert-pretraining/datasets/enwiki/enwiki-feb-doc-split/*.dev' \
                --num_steps 14063 --max_seq_length 128 --lr 0.005 --warmup_ratio 0.2 \
                --total_batch_size 32768 --no_compute_acc  --max_predictions_per_seq 20 \
                --total_batch_size_eval 32768 \
                --optimizer lamb2 \
                --ckpt_interval 99999999 \
                --raw --log_interval 50 --accumulate 2 \
                --ckpt_dir ./32K_np256_ckpt_dir_lamb2_eps_rescale_numworker 2>&1 | tee -a 32K_np256_ckpt_dir_lamb2_eps_rescale_numworker.log

        #-x HOROVOD_TIMELINE=timeline_tree.json \
                #--synthetic_data --eval_use_npz \
