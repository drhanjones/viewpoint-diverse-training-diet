from functools import partial
import json
import webdataset as wds
import paramiko
from typing import List, Tuple
from pathlib import Path
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import gzip
from webdataset import split_by_node
from concurrent.futures import ProcessPoolExecutor
import torch
from tqdm.auto import tqdm

PathLike = str | Path


class RemoteWebDatasetLoader:
    """Data loader for WebDataset format."""

    # TODO: Update later
    remote_dir = "/home/athamma1/project_storage_shortcut/outputs/diverse_objects_wds"
    client: paramiko.SSHClient

    def __init__(self, host: str, username: str):
        self.host = host
        self.username = username

        self.urls = self.list_remote_shard_urls(host, username)
        self.dataset = (
            wds.WebDataset(self.urls, shardshuffle=1000)
            .shuffle(1000)
            .decode("pil")
            .to_tuple("png", "metadata.json")
        )

    def _connect_sftp(self, host: str, username: str) -> paramiko.SFTPClient:
        print("Connecting to remote host via SFTP")
        client = paramiko.SSHClient()
        client.load_system_host_keys()

        client.connect(
            hostname=host,
            username=username,
            look_for_keys=True,
            allow_agent=True,
            timeout=30,
        )
        self.client = client
        return client.open_sftp()

    def list_shards_sftp(
        self,
        host: str,
        username: str,
        remote_dir: str,
        exts: Tuple[str, ...] = (".tar", ".tar.gz", ".tgz"),
    ) -> List[str]:
        """Return full remote paths to shard files (unsorted -> sorted)."""
        sftp = self._connect_sftp(host, username)
        print(f"Connected to host")

        print("Listing shards in remote dir")
        try:
            names = sftp.listdir(remote_dir)
            shards = [f"{remote_dir}/{n}" for n in names if n.lower().endswith(exts)]
            # sort for reproducibility; you can customize (mtime sort, natural sort, etc.)
            shards.sort()
            return shards
        finally:
            try:
                sftp.close()
            except Exception:
                pass

    def list_remote_shard_urls(self, host: str, username: str):
        user_host = username + "@" + host
        shard_paths = self.list_shards_sftp(host, username, self.remote_dir)
        urls_list = [
            f"pipe:ssh {user_host} 'cat {shard_path}'" for shard_path in shard_paths
        ]

        return urls_list


class WebDatasetLoader:
    def __init__(
        self,
        dataset_path: Path,
        key_to_category_mapper_path: Path,
        train_val_split: float = 0.8,
        seed: int = 42,
        batch_size: int = 32,
        build_dataloaders: bool = True,
        test_setup: bool = False,
    ):
        self.dataset_path: Path = dataset_path
        url_paths: List[Path] = sorted(self.dataset_path.glob("*.tar"))
        self.urls: List[str] = ["file://" + str(url) for url in url_paths]
        if test_setup:
            self.urls = self.urls[:60]

        self.train_urls, self.val_urls = self.shuffle_and_split_dataset(
            train_val_split, seed
        )
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.key_to_category_mapper, self.label_encoder = self.build_label_encoder(
            key_to_category_mapper_path=key_to_category_mapper_path
        )
        self.label_to_idx = {
            label: idx for idx, label in enumerate(self.label_encoder.classes_)
        }

        if build_dataloaders:
            self.build_dataloaders(batch_size, num_workers=8)

    @staticmethod
    def build_sample(sample, transforms, key_to_category_mapper, label_to_idx):
        sample_image, metadata_dict, sample_key = sample

        x = transforms(sample_image)
        # print(f"Sample key: {sample_key}, Label: {repr(self.key_to_category_mapper.get(sample_key))}")
        y = key_to_category_mapper.get(sample_key)
        if y is None:
            y = "basket"  # Temporary fix for bug in data for basket class
        y = label_to_idx.get(y, 0)  # Some bug in data for basket class, temporary fix

        return x, y

    def build_dataloaders(self, batch_size: int, num_workers: int):

        train_workers = min(num_workers, len(self.train_urls))
        val_workers = min(num_workers, len(self.val_urls))

        build_fn = partial(
            self.build_sample,
            transforms=self.transforms,
            key_to_category_mapper=self.key_to_category_mapper,
            label_to_idx=self.label_to_idx,
        )

        train_dataset = (
            wds.WebDataset(self.train_urls, shardshuffle=False)  # type: ignore
            .shuffle(1000)
            .decode("pil")
            .to_tuple("png", "metadata.json", "__key__")
            .map(build_fn)
            .batched(batch_size)
        )

        val_dataset = (
            wds.WebDataset(self.val_urls, shardshuffle=False)  # type: ignore
            .decode("pil")
            .to_tuple("png", "metadata.json", "__key__")
            .map(build_fn)
            .batched(batch_size)
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=None,
            pin_memory=True,
            num_workers=train_workers,
            multiprocessing_context="fork",
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=None,
            pin_memory=True,
            num_workers=val_workers,
            multiprocessing_context="fork",
        )

    def shuffle_and_split_dataset(
        self, train_val_split: float = 0.8, seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        shuffled_urls = self.urls.copy()
        rng = random.Random(seed)
        rng.shuffle(shuffled_urls)
        split_index = int(len(shuffled_urls) * train_val_split)
        print(
            f"Total shards: {len(shuffled_urls)}, Train shards: {split_index}, Val shards: {len(shuffled_urls) - split_index}"
        )
        train_urls = shuffled_urls[:split_index]
        print(f"First train shard: {train_urls[0]}, Last train shard: {train_urls[-1]}")
        val_urls = shuffled_urls[split_index:]
        print(f"First val shard: {val_urls[0]}, Last val shard: {val_urls[-1]}")
        return train_urls, val_urls

    def build_label_encoder(
        self, key_to_category_mapper_path: Path
    ) -> Tuple[dict, LabelEncoder]:
        with gzip.open(key_to_category_mapper_path, "rt") as f:
            key_to_category_mapper = json.load(f)

        categories = list(set(key_to_category_mapper.values()))
        print(f"Categories before cleaning: {categories}")
        categories = [x for x in categories if x is not None and x != "None"]
        print(f"Categories after cleaning: {categories}")
        categories = sorted(categories)

        label_encoder = LabelEncoder()
        label_encoder.fit(categories)

        return key_to_category_mapper, label_encoder

    def get_num_classes(self) -> int:
        return len(self.label_encoder.classes_)


class DistributedWebDatasetLoader(WebDatasetLoader):
    def __init__(
        self,
        dataset_path: Path,
        key_to_category_mapper_path: Path,
        rank: int,
        world_size: int,
        train_val_split: float = 0.8,
        seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 4,
        test_setup: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size

        super().__init__(
            dataset_path=dataset_path,
            key_to_category_mapper_path=key_to_category_mapper_path,
            train_val_split=train_val_split,
            seed=seed,
            batch_size=batch_size,
            build_dataloaders=False,
            test_setup=test_setup,
        )

        estimated_train_size = count_samples_multiprocess(
            self.train_urls[self.rank :: self.world_size], max_workers=16
        )
        estimated_val_size = count_samples_multiprocess(
            self.val_urls[self.rank :: self.world_size], max_workers=16
        )
        estimated_sizes = torch.tensor(
            [estimated_train_size, estimated_val_size],
            dtype=torch.long,
            device=f"cuda:{self.rank}",
        )
        torch.distributed.all_reduce(estimated_sizes, op=torch.distributed.ReduceOp.SUM)
        estimated_train_size = int(estimated_sizes[0].item())
        estimated_val_size = int(estimated_sizes[1].item())

        print(
            f"Rank {self.rank}: Estimated train samples: {estimated_train_size}, Estimated val samples: {estimated_val_size}"
        )

        self.build_dataloaders_with_rank(
            batch_size,
            num_workers,
            total_sample_count=(estimated_train_size, estimated_val_size),
        )

        if self.rank == 0:
            print(
                f"DDP: {self.world_size} ranks, {len(self.train_urls)} train shards, {len(self.val_urls)} val shards"
            )

    def shuffle_and_split_dataset(self, train_val_split: float = 0.8, seed: int = 42):
        """Override to split by rank after train/val split"""
        train_urls, val_urls = super().shuffle_and_split_dataset(train_val_split, seed)
        return train_urls, val_urls

    def get_shards_for_rank(self, urls: List[str]) -> List[str]:
        """Distribute shards among ranks as evenly as possible."""
        if self.world_size == 1:
            return urls

        rank_urls = urls[self.rank :: self.world_size]

        return rank_urls

    def build_dataloaders_with_rank(
        self, batch_size: int, num_workers: int, total_sample_count: Tuple[int, int]
    ):
        """Rebuild dataloaders with num_workers and proper batching"""

        train_workers = min(num_workers, len(self.train_urls) // self.world_size)
        val_workers = min(num_workers, len(self.val_urls) // self.world_size)

        build_fn = partial(
            self.build_sample,
            transforms=self.transforms,
            key_to_category_mapper=self.key_to_category_mapper,
            label_to_idx=self.label_to_idx,
        )

        train_dataset = (
            wds.WebDataset(
                self.train_urls, shardshuffle=True, nodesplitter=split_by_node
            )  # type: ignore
            .shuffle(1000)
            .decode("pil")
            .to_tuple("png", "metadata.json", "__key__")
            .map(build_fn)
            .batched(batch_size)
        )

        self.train_dataloader = wds.WebLoader(
            train_dataset,
            batch_size=None,
            num_workers=train_workers,
            pin_memory=True,
            persistent_workers=train_workers > 0,
        ).with_epoch(total_sample_count[0] // (batch_size * self.world_size))

        # Val loader
        val_dataset = (
            wds.WebDataset(
                self.val_urls, shardshuffle=False, nodesplitter=split_by_node
            )  # type: ignore
            .decode("pil")
            .to_tuple("png", "metadata.json", "__key__")
            .map(build_fn)
            .batched(batch_size)
        )

        self.val_dataloader = wds.WebLoader(
            val_dataset,
            batch_size=None,
            num_workers=val_workers,
            pin_memory=True,
            persistent_workers=val_workers > 0,
        ).with_epoch(total_sample_count[1] // (batch_size * self.world_size))


def count_samples_in_shard(path: str) -> int:
    ds = wds.WebDataset(
        path, nodesplitter=lambda src: src, shardshuffle=False
    ).to_tuple("__key__")
    return sum(1 for _ in ds)


def count_samples_multiprocess(shard_paths: list[str], max_workers: int = 16) -> int:
    """Use processes—good for local SSD (avoids GIL)."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        counts = list(
            tqdm(
                executor.map(count_samples_in_shard, shard_paths),
                total=len(shard_paths),
                desc="Counting samples in shards",
            )
        )
    return sum(counts)
