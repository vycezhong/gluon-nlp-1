
#clush --hostfile $OTHER_HOST "ls ~/efs"

clush -l $CLUSHUSER --hostfile $OTHER_HOST "docker pull $DOCKER_IMAGE"

#clush --hostfile $OTHER_HOST "ls ~/ssh_info;"

clush -l $CLUSHUSER --hostfile $OTHER_HOST "mkdir -p ~/haibin/tmp/ckpts; df -h ~/haibin/tmp/ckpts;"
clush -l $CLUSHUSER --hostfile $OTHER_HOST "sudo rm -rf ~/ssh_info; cp -r ~/.ssh ~/ssh_info;"
clush -l $CLUSHUSER --hostfile $OTHER_HOST 'docker kill $(docker ps -q);'

#clush --hostfile $OTHER_HOST "docker ps";
clush -l $CLUSHUSER --hostfile $OTHER_HOST "docker ps --no-trunc";
clush -l $CLUSHUSER --hostfile $OTHER_HOST "nvidia-docker run -d --security-opt seccomp:unconfined --privileged  \
                 -v ~/ssh_info:/root/.ssh  \
                 -v /home/$CLUSHUSER/mxnet-data/bert-pretraining/datasets:/data              \
                 -v /home/$CLUSHUSER/efs/haibin:/generated                                   \
                 -v /home/$CLUSHUSER/haibin/tmp/ckpts:/bert/ckpts                            \
                 --network=host --shm-size=32768m --ulimit nofile=65536:65536 $DOCKER_IMAGE  \
                 bash -c 'bash hvd_ssh.sh; cd gluon-nlp/; git fetch; git reset --hard $COMMIT; /usr/sbin/sshd -p $PORT -d'"
clush -l $CLUSHUSER --hostfile $OTHER_HOST "docker ps --no-trunc";
