import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from viewpoint_diverse_training_diet.data_loader.data_loaders import BaseDataLoader
from viewpoint_diverse_training_diet.utils.util import seed_worker


class DistributedBaseDataLoader(BaseDataLoader):
    """BaseDataLoader extension with DistributedSampler support."""

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        *,
        distributed=None,
        collate_fn=default_collate,
        pin_memory=False,
        drop_last=False,
    ):
        distributed = distributed or {}
        self.distributed_cfg = distributed
        self.is_distributed = distributed.get('enabled', False)
        self.rank = distributed.get('rank', 0)
        self.world_size = distributed.get('world_size', 1)

        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0

        self.dataset, self.valid_dataset = self._split_dataset(dataset, validation_split)
        self.n_samples = len(self.dataset)

        self.init_kwargs = {
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last,
            'worker_init_fn': seed_worker,
        }

        if self.is_distributed:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=self.shuffle,
            )
            shuffle_flag = False
        else:
            self.sampler = None
            shuffle_flag = self.shuffle

        DataLoader.__init__(
            self,
            dataset=self.dataset,
            sampler=self.sampler,
            shuffle=shuffle_flag,
            **self.init_kwargs,
        )

    def _split_dataset(self, dataset, split):
        if split == 0.0:
            return dataset, None

        indices = np.random.permutation(len(dataset))

        if isinstance(split, int):
            assert 0 < split < len(dataset), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(len(dataset) * split)

        valid_idx = indices[:len_valid]
        train_idx = indices[len_valid:]

        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        return train_dataset, valid_dataset

    def split_validation(self):
        if self.valid_dataset is None:
            return None

        if self.is_distributed:
            sampler = DistributedSampler(
                self.valid_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        else:
            sampler = None

        return DataLoader(
            dataset=self.valid_dataset,
            sampler=sampler,
            shuffle=(sampler is None),
            batch_size=self.init_kwargs['batch_size'],
            num_workers=self.init_kwargs['num_workers'],
            pin_memory=self.init_kwargs['pin_memory'],
            drop_last=False,
            collate_fn=self.init_kwargs['collate_fn'],
            worker_init_fn=self.init_kwargs['worker_init_fn'],
        )


class DistributedMnistDataLoader(DistributedBaseDataLoader):
    """MNIST data loader with optional distributed sampling."""

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
        *,
        distributed=None,
        collate_fn=default_collate,
        pin_memory=False,
        drop_last=False,
    ):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            distributed=distributed,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )


# Alias matching the base loader name for config compatibility
MnistDataLoader = DistributedMnistDataLoader
