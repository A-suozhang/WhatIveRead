"""
Example:
Start the ray head on 4 gpus: 0, 2, 3, 4
`CUDA_VISIBLE_DEVICES=0,2,3,4 ray start --head --num-cpus 10 --num-gpus 4 --object-store-memory 10000000`

Then, for every search-derive-sur-final flow, run:
`python ray_pipeline.py --redis-addr <head redis server address> <other args>`

The redis server address of ray head will be printed by the `ray start` command.

..warning::
Currently, all remote tasks write to the filesystem directly, so if there are multiple nodes used,
make sure all these files are on a shared filesystem between these nodes.

Suppose there is a new node with a shared filesystem, to add this node into the ray cluster:
`CUDA_VISIBLE_DEICES=1,2,4,5,6 ray start --redis-address <head redis server address> --num-cpus 10 --num-gpus=5 --object-store-memory 10000000`
"""

import argparse
import time
import os
import sys
import re
import glob
import signal

import random
import psutil
from multiprocessing import Process
import subprocess

import ray
def _get_gpus(gpu_ids):
    return ",".join(map(str, gpu_ids))

class KillSignal(ray.experimental.signal.Signal):
    pass

@ray.remote
class Killer(object):
    def send_kill(self):
        ray.experimental.signal.send(KillSignal())
        print("finished sending kill signals, please wait for some seconds for all these tasks to exit")

@ray.remote(num_gpus=1, max_calls=1)
def call_train(cfg, seed, train_dir, save_every, killer):
    if seed is None:
        seed = random.randint(1, 999999)
    print("train seed: %s" % str(seed))
    gpus = ray.get_gpu_ids()
    gpu = _get_gpus(gpus)
    save_str = "" if save_every is None else "--save-every {}".format(save_every)

    # setup gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # call aw_nas.main::main
    cmd = ("{cfg} --gpu {range_gpu}"
           " --seed {seed} {save_str} --save {train_dir}")\
        .format(cfg=cfg, seed=seed, train_dir=train_dir, save_str=save_str,
                range_gpu=",".join(map(str, range(len(gpus)))))
    print("CUDA_VISIBLE_DEVICES={} {}".format(gpu, cmd))
    print("check {} for log".format(os.path.join(train_dir, "train.log")))
    def _run_main(*args):
        sys.stdout = open("/dev/null", "w")
        from main import main
        main(*args)
    proc = Process(target=_run_main, args=(re.split(r"\s+", cmd),))
    proc.start()

    # wait for proc finish or killed
    while 1:
        time.sleep(10)
        if proc.is_alive():
            sigs = ray.experimental.signal.receive([killer], timeout=1)
            if sigs:
                print("call_train: receive kill signal from killer, kill the working processes")
                process = psutil.Process(proc.pid)
                for c_proc in process.children(recursive=True):
                    c_proc.kill()
                process.kill()
                exit_status = 1
                break
        else:
            exit_status = proc.exitcode
            break
    if exit_status != 0:
        raise subprocess.CalledProcessError(exit_status, cmd)

    return os.path.join(train_dir, "train.log")

def terminate_procs(sig, frame):
    print("sending kill signals, please wait for some seconds for all these tasks to exit")
    killer.send_kill.remote()

signal.signal(signal.SIGINT, terminate_procs)
signal.signal(signal.SIGTERM, terminate_procs)

parser = argparse.ArgumentParser()
parser.add_argument("--redis-addr", required=True, type=str,
                    help="the redis server address of ray head")
parser.add_argument("--result-base-dir", required=True, type=str,
                    help="the result base directory")
cmd_args, dir_names = parser.parse_known_args()
print("dir names: ", dir_names)

ray.init(redis_address=cmd_args.redis_addr) # connect to ray head


result_base_dir = cmd_args.result_base_dir

killer = Killer.remote() # create the killer actor

results = []
for dir_name in dir_names:
    yaml_files = glob.glob(os.path.join(dir_name, "*.yaml"))
    dir_basename = os.path.basename(os.path.abspath(dir_name))
    for yaml_file in yaml_files:
        label = os.path.basename(yaml_file).rsplit(".", 1)[0]
        train_dir = os.path.join(result_base_dir, dir_basename, label)
        print("config file: {}; save to {}".format(yaml_file, train_dir))
        results.append(call_train.remote(yaml_file, seed=123,
                                         train_dir=train_dir,
                                         save_every=40,
                                         killer=killer
                                         ))
try:
    results = ray.get(results)
except ray.exceptions.RayTaskError as e:
    print("EXIT! Exception when executing task: ", e)
    sys.exit(1)
