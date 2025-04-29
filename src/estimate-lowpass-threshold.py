"""
This is the script used to train SAEs on each layer of the ChessGPT2 model
"""

import argparse
import os

import dotenv
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from tqdm.auto import tqdm

from src.callbacks import NtfyCallback, WandbLogger
from src.data import SaeDataModule, SingleHiddenStateCollator
from src.model import LightSparseAutoEncoder
from src.sae import SparseAutoEncoder, ThresholdFilter
from src.utils.ntfy import Ntfy

torch.set_float32_matmul_precision("medium")


class ModelCheckpointPlus(ModelCheckpoint):

    def __init__(self, light_sae: LightSparseAutoEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.light_sae = light_sae

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer=trainer, filepath=filepath)
        self.light_sae.sae.save_pretrained(os.path.splitext(filepath)[0])


def main():
    dotenv.load_dotenv()
    # fmt: off
    parser = argparse.ArgumentParser("Train SAE on a single layer")
    # Training params
    parser.add_argument("--layer", type=int, default=9, help="The layer on which to train the SAE.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Number of hidden state vectors in each training batch.")
    parser.add_argument("--num_workers", type=int, default=15, help="Number of workers used by the torch 🔥 Dataloader.")
    parser.add_argument("--num_proc", type=int, default=3, help="Number of processes used by the Huggingface 🤗 Dataset module.")
    # fmt: on

    args = parser.parse_args()

    ############################
    # Setup Datamodule
    ############################

    dm = SaeDataModule(
        layer=args.layer,
        collator=SingleHiddenStateCollator(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
    )
    dm.setup()

    ############################
    # Setup Model
    ############################

    sae = SparseAutoEncoder.from_pretrained(f"models/saes/layer{args.layer:02d}").to("cuda")

    ############################
    # Run
    ############################
    min_vals = []
    with torch.no_grad():
        for record in tqdm(dm.train_dataloader(), desc=f"Ecoding"):
            encoding: torch.Tensor = sae.encode(record.to("cuda"))
            min_vals.append(encoding[encoding != 0].min().to("cpu"))

    lowpass_threshold = torch.stack(min_vals).mean()

    print(f"Found L{args.layer} lowpass filter threshold: {lowpass_threshold}")

    sae.config.sparsity_scheme = "threshold"
    sae.config.sparsity_parameter = lowpass_threshold.item()

    sae.save_pretrained(f"models/saes-inference/layer{args.layer:02d}")


if __name__ == "__main__":
    main()
