from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.datasets import DATASETS, build_dataloader

from tqdm import tqdm

import multiprocessing as mp
from multiprocessing import Process, Manager

def parse_args():
    parser = argparse.ArgumentParser(description='Debug polarmask target in pipeline')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    distributed = False
    datasets = [build_dataset(cfg.data.train)]
    error_indexs = []
    import pdb
    pdb.set_trace()
    for i in tqdm(range(len(datasets[0]))):
        try:
            _ = datasets[0][i]
        except:
            error_indexs.append(i)

    # import pdb
    # pdb.set_trace()

def dataloader_test():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    distributed = False
    datasets = [build_dataset(cfg.data.train)]
    data = datasets[0][40]
    import pdb
    pdb.set_trace()
    # data_loaders = [
    #     build_dataloader(
    #         ds,
    #         cfg.data.imgs_per_gpu,
    #         cfg.data.workers_per_gpu,
    #         1,
    #         dist=False) for ds in datasets
    # ]
    # for data in data_loaders[0]:
    #     pass

if __name__ == "__main__":
    #main()
    dataloader_test()