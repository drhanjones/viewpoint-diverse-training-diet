"""Distributed training utilities with safe fallbacks."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Dict

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank(default: int = 0) -> int:
    if not is_dist_avail_and_initialized():
        return default
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def init_distributed_mode(config: Dict) -> Dict:
    distributed_cfg = config.get('distributed', {}) or {}
    if not distributed_cfg.get('enabled', False):
        distributed_cfg['enabled'] = False
        config['distributed'] = distributed_cfg
        return distributed_cfg

    if not dist.is_available():
        distributed_cfg['enabled'] = False
        config['distributed'] = distributed_cfg
        return distributed_cfg

    backend = distributed_cfg.get('backend', 'nccl')
    init_method = distributed_cfg.get('init_method', 'env://')

    if init_method.startswith('env://'):
        rank = int(os.environ.get('RANK', distributed_cfg.get('rank', 0)))
        world_size = int(os.environ.get('WORLD_SIZE', distributed_cfg.get('world_size', 1)))
        local_rank = int(os.environ.get('LOCAL_RANK', distributed_cfg.get('local_rank', 0)))
    else:
        rank = distributed_cfg.get('rank', 0)
        world_size = distributed_cfg.get('world_size', 1)
        local_rank = distributed_cfg.get('local_rank', rank)

    if distributed_cfg.get('auto_set_device', True) and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)

    barrier()

    distributed_cfg.update({
        'enabled': True,
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'backend': backend,
        'init_method': init_method,
    })
    config['distributed'] = distributed_cfg
    return distributed_cfg


def reduce_dict(input_dict: Dict[str, float], average: bool = True) -> Dict[str, float]:
    if not is_dist_avail_and_initialized():
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.tensor([input_dict[k] for k in keys], device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float64)
        dist.all_reduce(values)
        if average:
            values /= get_world_size()
        return {k: values[idx].item() for idx, k in enumerate(keys)}


def sync_seed(seed: int) -> int:
    if not is_dist_avail_and_initialized():
        return seed

    seed_tensor = torch.tensor(seed, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.int32)
    dist.broadcast(seed_tensor, src=0)
    return int(seed_tensor.item())
