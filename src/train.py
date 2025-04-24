print("Script started")
import argparse
import os

import dotenv
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner

from src.callbacks import NtfyCallback, WandbLogger
from src.data import SaeDataModule, SingleLayerHiddenStateCollator
from src.losses import L1WeightedLoss, MSELoss
from src.model import SparseAutoEncoder
from src.sparsity import BatchTopKFilter, TopKFilter
from src.utils.ntfy import Ntfy

torch.set_float32_matmul_precision("medium")


def main():
    dotenv.load_dotenv()
    # fmt: off
    parser = argparse.ArgumentParser("Train SAE on a single layer")
    parser.add_argument("--layer", type=int, default=0, help="The layer on which to train the SAE.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Number of hidden state vectors in each training batch.")
    parser.add_argument("--lr", type=float, default=0.001, help="Number of hidden state vectors in each training batch.")
    parser.add_argument("--num_workers", type=int, default=15, help="Number of workers used by the torch 🔥 Dataloader.")
    parser.add_argument("--num_proc", type=int, default=3, help="Number of processes used by the Huggingface 🤗 Dataset module.")
    parser.add_argument("--dict_size", type=int, default=None, help="Explicit size of the dictionary (See --dict_scale).")
    parser.add_argument("--dict_scale", type=int, default=4, help="Multiplier for the hidden state dimension to compute the dictionary size when --dict_size is not provided.")
    parser.add_argument("--activation_dim", type=int, default=768, help="Size of the input/output vector data. (default: 768)")
    parser.add_argument("--k", type=int, default=32, help="k parameter of the top-k and batch top-k activation functions")
    parser.add_argument("--activation", choices=["topk", "batch_topk"], default="topk", help="Selects the activation function: 'topk' for TopKFilter, 'batch_topk' for BatchTopKFilter.")
    parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights and Biases")
    parser.add_argument("--ntfy", action="store_true", help="Enable Ntfy notifications callback")
    parser.add_argument("--checkpoint", action="store_true", help="Enable model checkpointing")
    parser.add_argument("--fast_dev_run", action="store_true", help="Runs 1 batch if set to True batch(es) of train, val and test to find any bugs (ie: a sort of unit test)")
    parser.add_argument("--find_batch_size", action="store_true", help="Enable Tuner to find batch_size")
    parser.add_argument("--find_lr", action="store_true", help="Enable Tuner to find learning rate (lr)")
    parser.add_argument("--train_loss_fn", choices=["mse", "l1"], default="l1", help="Select the loss function to use during training either 'mse' for MSELoss() or 'l1' for L1WeightedLoss().")
    parser.add_argument("--l1_coeff", type=float, default=1.0, help="The l1_coefficient to use by the L1WeightedLoss.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of training epochs to run.")
    # fmt: on

    args = parser.parse_args()

    ############################
    # Setup Datamodule
    ############################

    dm = SaeDataModule(
        layer=args.layer,
        collator=SingleLayerHiddenStateCollator(layer=args.layer),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
    )

    ############################
    # Setup Model
    ############################
    dict_size = args.dict_size if args.dict_size is not None else args.dict_scale * 768
    activation_cls = {"topk": TopKFilter, "batch_topk": BatchTopKFilter}[args.activation]
    activation_filter = activation_cls(k=args.k)
    train_loss_fn = MSELoss() if args.train_loss_fn == "mse" else L1WeightedLoss(args.l1_coeff)

    sae = SparseAutoEncoder(
        activation_dim=args.activation_dim,
        dict_size=dict_size,
        train_loss_fn=train_loss_fn,
        activation_fn=activation_filter,
        initial_learning_rate=args.lr,
    )

    ############################
    # Setup Trainer
    ############################

    callbacks = []

    if args.ntfy:
        callbacks.append(NtfyCallback(os.environ.get("NTFY_TOPIC", None)))

    if args.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath="models",
                monitor="validation/loss",  # metric to monitor
                mode="min",  # minimize val_acc
                save_top_k=2,  # keep only top 2 checkpoints
                save_last=True,  # also save the last checkpoint
                every_n_epochs=1,
                filename="epoch-{epoch}-step-{step}-{val_loss:.4f}",  # custom filename
            )
        )

    trainer = Trainer(
        accelerator="auto",
        fast_dev_run=args.fast_dev_run,
        limit_test_batches=0,
        # limit_val_batches=0,
        max_epochs=args.max_epochs,
        val_check_interval=1.0,  # When using an IterableDataset you must set the val_check_interval to 1.0
        callbacks=callbacks,
        logger=(
            WandbLogger(
                name=f"SAE-{args.layer:02d}-{dict_size}-{args.activation}-{args.k}",
                project="TrainSae",
                log_model=False,
                checkpoint_name=None,
            )
            if args.wandb
            else None
        ),
    )

    ############################
    # Setup Tuner
    ############################

    if args.find_batch_size or args.find_lr:
        tuner = Tuner(trainer)
        if args.find_lr:
            initial_lr = tuner.lr_find(model=sae, datamodule=dm)
            if args.ntfy:
                ntfy = Ntfy(topic=os.environ.get("NTFY_TOPIC", None))
                ntfy.send_notification(f"Training with initial_lr: {initial_lr.suggestion()=}")
        if args.find_batch_size:
            batch_size = tuner.scale_batch_size(model=sae, datamodule=dm, max_trials=10)
            if args.ntfy:
                ntfy = Ntfy(topic=os.environ.get("NTFY_TOPIC", None))
                ntfy.send_notification(f"Training with batch size: {batch_size=}")

    ############################
    # Run
    ############################
    trainer.fit(model=sae, datamodule=dm)


if __name__ == "__main__":
    main()
