#!/bin/bash

i=0
while true; then 
    var=`nvidia-smi --query-gpu=utilization.gpu --format=csv | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' `
    var=$(echo $var)
    var=$((${var// /+}))
    processes=`ps -ef|grep python | wc -l`

    if [[ $var -eq 0 ]]; then 
        i=$((i+1))
    else
        i=0
    fi

    if [[ i -eq 3 ]]; then
        echo date, "hang detected!"
        ./bps_v1.sh
        i=0
    fi

    sleep 60
done