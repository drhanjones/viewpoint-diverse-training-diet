import json
import os
import random
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import uuid


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def seed_everything(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def synchronize_between_processes(self):
        try:
            import torch.distributed as dist
        except ImportError:  # pragma: no cover - optional dependency
            return

        if not (dist.is_available() and dist.is_initialized()):
            return

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        for key in self._data.index:
            totals = torch.tensor(
                [self._data.total[key], self._data.counts[key]],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            self._data.total[key] = totals[0].item()
            self._data.counts[key] = max(totals[1].item(), 1.0)
            self._data.average[key] = self._data.total[key] / self._data.counts[key]


def get_job_unique_id():
    """Get a unique identifier for the current job, useful for logging and checkpointing."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        return job_id
    else:
        if os.environ.get("LOCAL_JOB_UUID") is not None:
            return os.environ["LOCAL_JOB_UUID"]
        else:
            new_uuid = str(uuid.uuid4())
            os.environ["LOCAL_JOB_UUID"] = new_uuid
            return new_uuid

def build_names(experiment_name_root="experiment", model_name="model"):
    """Build standard names for logging and checkpointing."""
    job_unique_id = get_job_unique_id()
    experiment_name = f"{experiment_name_root}_{model_name}_{job_unique_id}"
    checkpoint_path = f"checkpoints/{experiment_name}.pth"

    return {
        "experiment_name": experiment_name,
        "checkpoint_path": checkpoint_path,
    }



    