#!/usr/bin/env python3
import sys
import subprocess
import json
import os
import shlex
import logging
import numpy

import gridbench


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')


def execute(cmd, **kwargs):
    if 'stdout' not in kwargs:
        kwargs['stdout'] = subprocess.PIPE
    if 'stderr' not in kwargs:
        kwargs['stderr'] = subprocess.PIPE
    if 'check' not in kwargs:
        kwargs['check'] = True
    try:
        logging.info('Running cmd {} with subprocess.run args {}'.format(cmd, kwargs))
        process = subprocess.run(shlex.split(cmd), **kwargs)
    except subprocess.CalledProcessError as exc:
        logged_stdout = None
        try:
            logged_stdout = exc.stdout
        except Exception:
            pass

        logged_stderr = None
        try:
            logged_stderr = exc.stderr
        except Exception:
            pass

        logging.exception('Command {} with STDOUT [{}], STDERR [{}] got exception:'.format(cmd, logged_stdout, logged_stderr))
        raise exc

    return process.stdout, process.stderr, process.returncode


def integer_linear_space(fr, to, cnt):
    return list(map(lambda num: int(round(num)),
                    numpy.arange(fr, to + 1, (to - fr)/(cnt - 1))
               ))


def print_cmd(cmd):
    print("local=1 meta='{}' {}".format(os.getenv("meta"), cmd))


def main():
    machine = sys.argv[1]
    binary_name = "buck-out/gen/db#release"
    fixed_flags = "-c 127.0.0.1:3148 -n host1 -w ycsb -Xcpu32 -Xmem32G -XYcsbContentionKey4 -XVHandleBatchAppend"

    handle_parallel_param_template = "-XVHandleParallel{}"
    yscb_skew_factor_param_template = "-XYcsbSkewFactor{}"
    core_scaling_param_template = "-XCoreScaling{}"

    search_space = list(set(integer_linear_space(2, 20, 5) +
                   integer_linear_space(20, 100, 6)))
    search_space = sorted(search_space)


    binary = "buck-out/gen/db#release"

    os.environ["local"] = "1"
    for skewness in [30, 60, 90]:
        for para in search_space:
            for core in search_space:
                baked_varied_params = "{} {} {}".format(
                    yscb_skew_factor_param_template.format(skewness),
                    core_scaling_param_template.format(core),
                    handle_parallel_param_template.format(para),
                )
                final_params = "{} {}".format(fixed_flags, baked_varied_params)
                os.environ["meta"] = json.dumps({
                    "YcsbSkewFactor": skewness,
                    "VHandleParallel": para,
                    "YcsbContentionKey": 4,
                    "CoreScaling": core,
                    "machine": machine,
                })
                cmd = "./run-job.sh {machine} {binary} {flags}".format(
                    machine=machine, binary=binary, flags=final_params)
                print_cmd(cmd)
                # run_cmd(#)


if __name__ == "__main__":
    main()
# local=1 ./run-job.sh c151 buck-out/gen/db#release -c 127.0.0.1:3148 -n host1 -w ycsb -Xcpu32 -Xmem32G -XYcsbContentionKey4 -XYcsbSkewFactor90 -XVHandleBatchAppend -XVHandleParallel5

