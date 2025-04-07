import os
from glob import glob
from typing import Callable

import datasets
import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader


class SingleLayerHiddenStateCollator:

    def __init__(self, layer: int, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def __call__(self, batch_BLPD: list[dict[str, torch.Tensor]]):
        """
        batch_BLPD comes in as a list of dict-records with shape Batch
        So BLPD is [Batch, dict[Tensor[Layer, Position, Dimension]]]
        HiddenState dimension is [Layer, Position, Dimension]"""
        hidden_states = [x["HiddenStates"] for x in batch_BLPD]
        max_len = max(t.shape[1] for t in hidden_states)
        padded = [
            F.pad(t, (0, 0, 0, max_len - t.shape[1]))  # pad dim=1 (seq dim) with zeros
            for t in hidden_states
        ]
        stacked = torch.stack(padded, dim=0)  # shape: (batch, L, P, D)
        print(f"{stacked.shape=}")
        exit()
        return stacked
        # print(f"{type(batch_BLPD)=}")
        # print(f"{len(batch_BLPD)=}")
        # print(f"{(batch_BLPD[0]['HiddenStates'].shape)=}")
        # return torch.stack(batch_BLPD[0]["HiddenStates"])  # TODO, remove the [0] index


class SaeDataModule(LightningDataModule):

    data_root: str
    collator: Callable
    batch_size: int
    num_workers: int
    num_proc: int
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

    def get_loader(self, stage: str):
        loader = DataLoader(
            self.hf_dataset[stage],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
        return loader

    def train_dataloader(self):
        return self.get_loader("train")

    def val_dataloader(self):
        return self.get_loader("val")

    def test_dataloader(self):
        return self.get_loader("test")
