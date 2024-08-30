import sys
import os
from omegaconf import OmegaConf
import torch.utils.data as data


from common.data import create_base_transforms, create_base_dataloader

from .video_dataset import *


def get_dataloader(args):
    """Set dataloader.

    Args:
        args (object): Args load from get_params function.
        split (str): One of ['train', 'test']
    """
    transform = create_base_transforms(args.transform_params)

    #dataset_cfg = getattr(args, split).dataset
    dataset_cfg = args.test.dataset
    dataset_params = OmegaConf.to_container(dataset_cfg.params, resolve=True)
    dataset_params['transform'] = transform

    _dataset = eval(dataset_cfg.name)(**dataset_params)

    _dataloader = create_base_dataloader(args, _dataset)

    return _dataloader
