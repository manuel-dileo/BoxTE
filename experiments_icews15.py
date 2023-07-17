#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'d'}])


def to_cmd(c, _path=None):
    if c['embedding_dim'] <= 500:
        command = f'PYTHONPATH=. python3 ../main.py '\
            f'--train_path ../datasets/ICEWS05-15/train.txt '\
            f'--valid_path ../datasets/ICEWS05-15/valid.txt '\
            f'--test_path ../datasets/ICEWS05-15/test.txt '\
            f'--learning_rate 0.001 --num_negative_samples=75 --loss_type=ce '\
            f'--batch_size=1 --metrics_batch_size=-1 --grad_accum 256 '\
            f'--validation_step=10 --neg_sampling_type=d --num_epochs=100 --print_loss_step=10 --model_variant=BoxTE '\
            f'--norm_embeddings --nb_timebumps=5 --use_r_factor --no_initial_validation --use_time_reg '\
            f'--embedding_dim {c["embedding_dim"]} --time_reg_weight {c["time_reg_weight"]} --time_reg_norm {c["time_reg_norm"]} '\
            f'--time_reg_order {c["time_reg_order"]}'
    else:
        command = f'PYTHONPATH=. python3 ../main.py ' \
            f'--train_path ../datasets/ICEWS05-15/train.txt ' \
            f'--valid_path ../datasets/ICEWS05-15/valid.txt ' \
            f'--test_path ../datasets/ICEWS05-15/test.txt ' \
            f'--learning_rate 0.001 --num_negative_samples=75 --loss_type=ce ' \
            f'--batch_size=1 --metrics_batch_size=-1 --grad_accum 256 ' \
            f'--validation_step=10 --neg_sampling_type=d --num_epochs=30 --print_loss_step=10 --model_variant=BoxTE ' \
            f'--norm_embeddings --nb_timebumps=5 --use_r_factor --no_initial_validation --use_time_reg ' \
            f'--embedding_dim {c["embedding_dim"]} --time_reg_weight {c["time_reg_weight"]} --time_reg_norm {c["time_reg_norm"]} ' \
            f'--time_reg_order {c["time_reg_order"]}'
    return command


def to_logfile(c, path):
    outfile = "{}/icews15.{}.log".format(path, summary(c).replace("/", "_")) #change here for other datasets
    return outfile


def main(argv):
    hyp_space = [
        dict(
            embedding_dim=[5, 25, 50, 100, 500],
            time_reg_weight=[1, 1e-1, 1e-2, 1e-3, 1e-4],
            time_reg_norm=['Lp', 'Np'],
            time_reg_order=[1, 2, 3, 4, 5],
        ),
        dict(
            embedding_dim=[2000],
            time_reg_weight=[1, 1e-1, 1e-2, 1e-3, 1e-4],
            time_reg_norm=['Np', 'Lp'],
            time_reg_order=[1, 2, 3, 4, 5],
        ),
    ]

    configurations = list(cartesian_product(hyp_space[int(argv[0])]))

    path = 'logs/icews15'
    path_from_here = 'scripts/logs/icews15'

    # If the folder that will contain logs does not exist, create it
    #if not os.path.exists(path):
        #os.makedirs(path)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)
        logfilefromhere = to_logfile(cfg, path_from_here)

        completed = False
        if os.path.isfile(logfilefromhere):
            with open(logfilefromhere, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'TEST METRICS' in content

        if not completed:
            cmd = to_cmd(cfg)
            if cmd is not None:
                command_line = f'{cmd} > {logfile} 2>&1'
                command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    try:
        is_slurm = eval(argv[1])
    except Exception:
        is_slurm = True

    header = None
    if is_slurm:
        header = f"""#!/usr/bin/env bash

#SBATCH --output=/home/%u/slogs/BoxTE-%A_%a.out
#SBATCH --error=/home/%u/slogs/BoxTE-%A_%a.err
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB # memory
#SBATCH --cpus-per-task=4 # number of cpus to use - there are 32 on each node.
#SBATCH -t 8:00:00 # time requested in hours:minutes:seconds
#SBATCH --array 1-{nb_jobs}

echo "Setting up bash environment"
source ~/.bashrc
set -e # fail fast

conda activate mypt

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/projects/BoxTE/scripts

"""

    if header is not None:
        print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        if is_slurm:
            print(f'test $SLURM_ARRAY_TASK_ID -eq {job_id} && sleep 10 && {command_line}')
        else:
            print(f'{command_line}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])