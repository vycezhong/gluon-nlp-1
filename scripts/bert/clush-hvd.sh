
#clush --hostfile $HOST "ls ~/efs"

clush -l $CLUSHUSER --hostfile $HOST "docker pull $DOCKER_IMAGE"

#clush --hostfile $HOST "ls ~/ssh_info;"

clush -l $CLUSHUSER --hostfile $HOST "cp -r ~/.ssh ~/ssh_info;"
clush -l $CLUSHUSER --hostfile ~/other_hosts 'docker kill $(docker ps -q);'

#clush --hostfile $HOST "docker ps";
clush -l $CLUSHUSER --hostfile $HOST "docker ps --no-trunc";
clush -l $CLUSHUSER --hostfile $HOST "nvidia-docker run -d --security-opt seccomp:unconfined --privileged  \
                 -v ~/ssh_info:/root/.ssh  \
                 -v /home/$CLUSHUSER/mxnet-data/bert-pretraining/datasets:/data          \
                 -v /home/$CLUSHUSER/efs/haibin:/generated                               \
                 -v /home/$CLUSHUSER/efs/haibin/bert:/bert
                 --network=host --shm-size=32768m --ulimit nofile=65536:65536 $DOCKER_IMAGE   \
                 bash -c 'bash hvd_ssh.sh; cd gluon-nlp/; git fetch; git reset --hard $COMMIT; /usr/sbin/sshd -p $PORT -d'"
clush -l $CLUSHUSER --hostfile $HOST "docker ps --no-trunc";
