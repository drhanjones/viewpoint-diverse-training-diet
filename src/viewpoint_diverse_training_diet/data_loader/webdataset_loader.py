from importlib.metadata import metadata
import json
import webdataset as wds
import paramiko
from typing import List, Tuple
from pathlib import Path
import random
from copy import deepcopy
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gzip


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
    ):
        self.dataset_path: Path = dataset_path
        self.urls: List[Path] | List[str] = sorted(self.dataset_path.glob("*.tar"))
        self.urls = ["file://" + str(url) for url in self.urls]

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

        self.train_dataset = (
            wds.WebDataset(self.train_urls, shardshuffle=False)
            .shuffle(1000)
            .decode("pil")
            .to_tuple("png", "metadata.json", "__key__")
            .map(self.build_sample)
        )

        self.val_dataset = (
            wds.WebDataset(self.val_urls, shardshuffle=False)
            .decode("pil")
            .to_tuple("png", "metadata.json", "__key__")
            .map(self.build_sample)
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            pin_memory=True,
        )

    def shuffle_and_split_dataset(
        self, train_val_split: float = 0.8, seed: int = 42
    ) -> Tuple[List[str] | List[Path], List[str] | List[Path]]:
        shuffled_urls = deepcopy(self.urls)

        rng = random.Random(seed)
        rng.shuffle(shuffled_urls)

        split_index = int(len(shuffled_urls) * train_val_split)
        train_urls = shuffled_urls[:split_index]
        val_urls = shuffled_urls[split_index:]
        return train_urls, val_urls

    # def collate_wds(self, batch, tf, metadata_dict, label_key="object_name"):
    #     """batch: list of (PIL_image, metadata_dict) from WebDataset"""
    #     images, metas = zip(*batch)
    #     x = torch.stack([tf(im.convert("RGB")) for im in images], dim=0)
    #     y = torch.tensor([metadata_dict[m[label_key]] for m in metas], dtype=torch.long)
    #     return x, y

    def build_label_encoder(
        self, key_to_category_mapper_path: Path
    ) -> Tuple[dict, LabelEncoder]:
        with gzip.open(key_to_category_mapper_path, "rt") as f:
            key_to_category_mapper = json.load(f)

        categories = list(set(key_to_category_mapper.values()))
        categories = sorted(categories)

        label_encoder = LabelEncoder()
        label_encoder.fit(categories)

        return key_to_category_mapper, label_encoder

    def build_sample(self, sample):
        sample_image, metadata_dict, sample_key = sample

        x = self.transforms(sample_image)
        y = self.key_to_category_mapper[sample_key]

        return x, y

    # def load_metadata_dictionary(
    #     self, metadata_path: Path
    # ) -> Tuple[dict, LabelEncoder]:
    #     metadata_df = pd.read_excel(
    #         metadata_path, engine="openpyxl", sheet_name="object_metadata_registry"
    #     )

    #     metadata_df = (
    #         metadata_df[["object_name", "category"]]
    #         .sort_values(by="category")
    #         .reset_index(drop=True)
    #     )

    #     label_encoder = LabelEncoder()
    #     metadata_df["label"] = label_encoder.fit_transform(metadata_df["category"])

    #     metadata_dict = dict(zip(metadata_df["object_name"], metadata_df["label"]))

    #     return metadata_dict, label_encoder
