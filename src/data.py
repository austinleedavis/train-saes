import os
from glob import glob
from typing import Callable

import datasets
import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader


class HiddenStateCollator:

    def __call__(self, batch_BLPD: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        batch_BLPD comes in as a list of dict-records with shape Batch
        BLPD is [Batch, dict[Tensor[Layer, Position, Dimension]]]
        HiddenState dimension is [Layer, Position, Dimension]"""
        hidden_states = [x["HiddenStates"] for x in batch_BLPD]
        max_len = max(t.shape[-2] for t in hidden_states)
        padded = [
            F.pad(
                t, (0, 0, 0, max_len - t.shape[-2]), mode="constant", value=0
            )  # pad dim=1 (num_pos dim) with zeros
            for t in hidden_states
        ]
        stacked = torch.stack(padded, dim=0)  # shape: (batch, L, P, D)

        attention_BP = torch.tensor(
            [[1] * t.shape[-2] + [0] * (max_len - t.shape[-2]) for t in hidden_states],
            dtype=torch.long,
        )

        attention_BLPD = attention_BP[:, None, :, None].expand(-1, 12, -1, 768)

        return stacked, attention_BLPD


class SingleHiddenStateCollator:

    def __call__(self, batch_BLD: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        batch_BLD comes in as a list of dict-records with shape Batch
        BLD is [Batch, dict[Tensor[Layer, Dimension]]]
        HiddenState dimension is [Layer, Dimension]"""
        hidden_states = [x["HiddenStates"] for x in batch_BLD]
        stacked_BLD = torch.stack(hidden_states, dim=0)  # shape: (batch, L, D)

        stacked_BLPD = stacked_BLD.unsqueeze(2)
        attention_BLPD = torch.ones_like(stacked_BLPD)

        return stacked_BLPD, attention_BLPD


class SaeDataModule(LightningDataModule):

    data_root: str
    collator: Callable
    batch_size: int
    num_workers: int
    """Number of works used by dataloaders"""
    num_proc: int
    """Number of processes used to load the dataset from disk"""
    hf_dataset: IterableDataset = None
    train_split: Dataset = None
    val_split: Dataset = None
    test_split: Dataset = None

    def __init__(
        self,
        data_root: str,
        collator: Callable,
        batch_size: int,
        num_workers: int,
        num_proc: int,
    ):
        super().__init__()
        self.data_root = data_root
        self.collator = collator
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_proc = num_proc

    def prepare_data(self) -> Dataset:
        pass

    def setup(self, stage: str = None):
        if not self.hf_dataset:
            for k in ["train", "test", "val"]:
                files = glob(os.path.join(self.data_root, k, "activations_*.parquet*"))
                print({k: f"count:{len(files)} first:{files[0]}"})

            self.hf_dataset = datasets.load_dataset(
                "parquet",
                data_files={
                    k: os.path.join(self.data_root, k, "activations_*.parquet*")
                    for k in ["train", "val", "test"]
                },
                num_proc=self.num_proc,
                # streaming=True,
            ).with_format("torch")

    def get_loader(self, stage: str, shuffle: bool):
        loader = DataLoader(
            self.hf_dataset[stage],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            shuffle=shuffle,
            drop_last=True,
        )
        return loader

    def train_dataloader(self):
        return self.get_loader("train", True)

    def val_dataloader(self):
        return self.get_loader("val", False)

    def test_dataloader(self):
        return self.get_loader("test", False)
