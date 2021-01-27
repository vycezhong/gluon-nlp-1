from pynvml.smi import nvidia_smi
from time import sleep
import os


nvsmi = nvidia_smi.getInstance()
out = nvsmi.DeviceQuery('utilization.gpu')
gpus = out['gpu']

cnt = 0
while True:
    total_util = 0
    for gpu in gpus:
        util = gpu['utilization']['gpu_util']
        total_util += util

    if total_util == 0:
        cnt += 1
    else:
        cnt = 0

    if cnt == 3:
        print("Hang detected. resume!")
        os.system("pkill python3; pkill bpslaunch")
        os.system("pwd")
        cnt = 0

    sleep(1)
