import os
from typing import Callable

import datasets
import torch
from datasets import Dataset, IterableDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader


class SingleLayerHiddenStateCollator:

    def __init__(self, layer: int, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def __call__(self, batch_BLPD: list[dict[str, torch.Tensor]]):
        """
        batch_BLPD comes in as a list of dict-records
        HiddenState dimension is [Batch, Layer, Position, Dimension]"""
        return batch_BLPD[0]["HiddenStates"][self.layer]


class SaeDataModule(LightningDataModule):

    hf_dataset: IterableDataset = None
    data_root: str = ""
    train_split: Dataset = None
    val_split: Dataset = None
    test_split: Dataset = None

    def __init__(
        self, data_root: str, collator: Callable, batch_size: int, num_workers: int
    ):
        super().__init__()
        self.data_root = data_root
        self.collator = collator
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> Dataset:
        pass

    def setup(self, stage: str = None):
        if not self.hf_dataset:
            self.hf_dataset = datasets.load_dataset(
                "parquet",
                data_files={
                    k: os.path.join(self.data_root, k, "*.parquet.gz")
                    for k in ["train", "val", "test"]
                },
                streaming=True,
            ).with_format("torch")

    def train_dataloader(self):
        loader = DataLoader(
            self.hf_dataset["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.hf_dataset["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.hf_dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
        return loader
