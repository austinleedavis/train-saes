import os
from glob import glob
from typing import Callable

import datasets
import torch
from datasets import Dataset, DatasetDict
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader


class SingleLayerHiddenStateCollator:

    def __init__(self, layer: int, target_column_name: str = "data", **kwargs):
        super().__init__(**kwargs)
        self.target_column_name = target_column_name

    def __call__(self, batch_BLPD: list[dict[str, torch.Tensor]]):
        """
        batch_BLPD comes in as a list of dict-records"""
        return torch.stack([b[self.target_column_name] for b in batch_BLPD])


class SaeDataModule(LightningDataModule):

    layer: int
    collator: Callable
    batch_size: int
    num_workers: int
    num_proc: int
    hf_dataset: Dataset = None
    train_split: Dataset = None
    val_split: Dataset = None
    test_split: Dataset = None

    def __init__(
        self,
        layer: int,
        collator: Callable,
        batch_size: int,
        num_workers: int,
        num_proc: int,
    ):
        super().__init__()
        self.layer = layer
        self.collator = collator
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_proc = num_proc

    def prepare_data(self) -> Dataset:
        pass

    def setup(self, stage: str = None):
        if not self.hf_dataset:
            phases = [0, 1, 2]
            configs = [f"layer-{self.layer:02d}-phase-{p}" for p in phases]

            ds: Dataset = (
                datasets.concatenate_datasets(
                    [
                        datasets.load_dataset(
                            path="austindavis/chessgpt2-hiddenstates",
                            name=config,
                            split="train",
                            num_proc=self.num_proc,
                        )
                        for config in configs
                    ]
                )
                .select_columns("data")
                .with_format("torch")
            )

            split = ds.train_test_split(test_size=0.1, seed=42)
            val_test = split["test"].train_test_split(test_size=0.5, seed=42)

            self.hf_dataset = DatasetDict(
                {
                    "train": split["train"],
                    "val": val_test["train"],
                    "test": val_test["test"],
                }
            )

    def get_loader(self, stage: str, shuffle=False):
        loader = DataLoader(
            self.hf_dataset[stage],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            shuffle=shuffle,
        )
        return loader

    def train_dataloader(self):
        return self.get_loader("train", shuffle=True)

    def val_dataloader(self):
        return self.get_loader("val")

    def test_dataloader(self):
        return self.get_loader("test")
