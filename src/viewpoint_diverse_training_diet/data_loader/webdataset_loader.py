import json
import webdataset as wds
import paramiko
from typing import List, Tuple
from pathlib import Path
import random
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import gzip
from webdataset import split_by_node


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
    ):
        self.dataset_path: Path = dataset_path
        url_paths: List[Path] = sorted(self.dataset_path.glob("*.tar"))
        self.urls: List[str] = ["file://" + str(url) for url in url_paths]

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

        if build_dataloaders:
            self.build_dataloaders(batch_size)

    def build_dataloaders(self, batch_size: int):
        train_dataset = (
            wds.WebDataset(self.train_urls, shardshuffle=False)  # type: ignore
            .shuffle(1000)
            .decode("pil")
            .to_tuple("png", "metadata.json", "__key__")
            .map(self.build_sample)
        )

        val_dataset = (
            wds.WebDataset(self.val_urls, shardshuffle=False)  # type: ignore
            .decode("pil")
            .to_tuple("png", "metadata.json", "__key__")
            .map(self.build_sample)
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
        )

    def shuffle_and_split_dataset(
        self, train_val_split: float = 0.8, seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        shuffled_urls = self.urls.copy()
        rng = random.Random(seed)
        rng.shuffle(shuffled_urls)
        split_index = int(len(shuffled_urls) * train_val_split)
        train_urls = shuffled_urls[:split_index]
        val_urls = shuffled_urls[split_index:]
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

    def build_sample(self, sample):
        sample_image, metadata_dict, sample_key = sample

        x = self.transforms(sample_image)
        # print(f"Sample key: {sample_key}, Label: {repr(self.key_to_category_mapper.get(sample_key))}")
        y = self.key_to_category_mapper.get(sample_key)
        if y is None:
            y = "basket"  # Temporary fix for bug in data for basket class
        y = self.label_encoder.transform([y])[
            0
        ]  # Some bug in data for basket class, temporary fix

        return x, y

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
        )

        self.build_dataloaders_with_rank(batch_size, num_workers)

        if self.rank == 0:
            print(
                f"DDP: {self.world_size} ranks, {len(self.train_urls)} train shards, {len(self.val_urls)} val shards"
            )

    def shuffle_and_split_dataset(self, train_val_split: float = 0.8, seed: int = 42):
        """Override to split by rank after train/val split"""
        train_urls, val_urls = super().shuffle_and_split_dataset(train_val_split, seed)

        # Split by rank
        train_urls = self.get_shards_for_rank(train_urls)
        val_urls = self.get_shards_for_rank(val_urls)

        return train_urls, val_urls

    def get_shards_for_rank(self, urls: List[str]) -> List[str]:
        """Distribute shards among ranks as evenly as possible."""
        if self.world_size == 1:
            return urls

        rank_urls = urls[self.rank :: self.world_size]

        return rank_urls

    def build_dataloaders_with_rank(self, batch_size: int, num_workers: int):
        """Rebuild dataloaders with num_workers and proper batching"""

        # Train loader
        train_dataset = (
            wds.WebDataset(self.train_urls, shardshuffle=True, nodesplitter=split_by_node)  # type: ignore
            .shuffle(1000)
            .decode("pil")
            .to_tuple("png", "metadata.json", "__key__")
            .map(self.build_sample)
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        # Val loader
        val_dataset = (
            wds.WebDataset(self.val_urls, shardshuffle=False, nodesplitter=split_by_node)  # type: ignore
            .decode("pil")
            .to_tuple("png", "metadata.json", "__key__")
            .map(self.build_sample)
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
