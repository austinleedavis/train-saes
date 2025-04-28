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

from src.callbacks import NtfyCallback, WandbLogger
from src.data import SaeDataModule, SingleLayerHiddenStateCollator
from src.model import LightSparseAutoEncoder
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
    parser.add_argument("--phases",type=int,nargs="+",help="List of integer phases")
    parser.add_argument("--layer", type=int, default=9, help="The layer on which to train the SAE.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Number of hidden state vectors in each training batch.")
    parser.add_argument("--num_workers", type=int, default=15, help="Number of workers used by the torch 🔥 Dataloader.")
    parser.add_argument("--num_proc", type=int, default=3, help="Number of processes used by the Huggingface 🤗 Dataset module.")
    parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights and Biases")
    parser.add_argument("--ntfy", action="store_true", help="Enable Ntfy notifications callback")
    parser.add_argument("--checkpoint", action="store_true", help="Enable model checkpointing")
    parser.add_argument("--early_stopping", action="store_true", help="Enable model early stopping")
    
    parser.add_argument("--fast_dev_run", action="store_true", help="Runs 1 batch if set to True batch(es) of train, val and test to find any bugs (ie: a sort of unit test)")
    parser.add_argument("--find_batch_size", action="store_true", help="Enable Tuner to find batch_size")
    parser.add_argument("--find_lr", action="store_true", help="Enable Tuner to find learning rate (lr)")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of training epochs to run.")
    
    #SAE params
    parser.add_argument("--dict_size", type=int, default=None, help="Explicit size of the dictionary (See --dict_scale).")
    parser.add_argument("--dict_scale", type=int, default=4, help="Multiplier for the hidden state dimension to compute the dictionary size when --dict_size is not provided.")
    parser.add_argument("--activation_dim", type=int, default=768, help="Size of the input/output vector data. (default: 768)")
    parser.add_argument("--pre_activation", type=str, choices=["relu", "none"], default="relu")
    parser.add_argument("--sparsity_scheme", type=str, choices=["topk", "batch_topk", "none"], default="batch_topk" , help="Selects the sparity activation function: 'topk' for TopKFilter, 'batch_topk' for BatchTopKFilter or 'none' if using L1WeightedLoss")
    parser.add_argument("--sparsity_parameter", type=int, default=32, help="The sparisty parameter used by the L1WeightedLoss or the (batch)topk")
    parser.add_argument("--train_loss_fn_str", type=str, choices=["mse", "weightedL1"], default="mse", help="Select the loss function to use during training either 'mse' for MSELoss() or 'l1' for L1WeightedLoss().")
    parser.add_argument("--val_loss_fn_str", type=str, choices=["mse", "weightedL1"], default="mse", help="Select the loss function to use during validation either 'mse' for MSELoss() or 'l1' for L1WeightedLoss().")
    parser.add_argument("--automatic_optimization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial_learning_rate", type=float, default=1e-3)
    # fmt: on

    args = parser.parse_args()

    ############################
    # Setup Datamodule
    ############################

    dm = SaeDataModule(
        layer=args.layer,
        phases=args.phases,
        collator=SingleLayerHiddenStateCollator(layer=args.layer),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
    )

    ############################
    # Setup Model
    ############################

    sae_module = LightSparseAutoEncoder(
        activation_dim=args.activation_dim,
        dict_size=args.dict_size,
        dict_scale=args.dict_scale,
        pre_activation=args.pre_activation,
        sparsity_scheme=args.sparsity_scheme,
        sparsity_parameter=args.sparsity_parameter,
        train_loss_fn_str=args.train_loss_fn_str,
        val_loss_fn_str=args.val_loss_fn_str,
        automatic_optimization=args.automatic_optimization,
        initial_learning_rate=args.initial_learning_rate,
    )

    ############################
    # Setup Trainer
    ############################

    callbacks = []

    if args.ntfy:
        callbacks.append(NtfyCallback(os.environ.get("NTFY_TOPIC", None)))

    if args.checkpoint:
        callbacks.append(
            ModelCheckpointPlus(
                light_sae=sae_module,
                dirpath="models",
                monitor="validation/loss",
                mode="min",
                save_top_k=1,
                # save_last=True,
                every_n_epochs=1,
                # filename="epoch-{epoch}-step-{step}-{val_loss:.4f}",  # custom filename
                filename=f"layer{args.layer:02d}",
            )
        )

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor="validation/loss", mode="min", patience=10, verbose=True))

    trainer = Trainer(
        accelerator="auto",
        fast_dev_run=args.fast_dev_run,
        max_epochs=args.max_epochs,
        val_check_interval=1.0,  # When using an IterableDataset you must set the val_check_interval to 1.0
        callbacks=callbacks,
        logger=(
            WandbLogger(
                name=f"SAE-{args.layer:02d}-{sae_module.sae.dict_size}-{args.sparsity_scheme}-{args.sparsity_parameter}",
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
            initial_lr = tuner.lr_find(model=sae_module, datamodule=dm)
            message = f"Training with initial_lr: {initial_lr.suggestion()=}"
            if args.ntfy:
                ntfy = Ntfy(topic=os.environ.get("NTFY_TOPIC", None))
                ntfy.send_notification(message)
            if args.wandb:
                import wandb

                wandb.log({"lr_find_results": wandb.Image(initial_lr.plot(suggest=True))})
            print(message)
        if args.find_batch_size:
            batch_size = tuner.scale_batch_size(model=sae_module, datamodule=dm, max_trials=10)
            message = f"Training with batch size: {batch_size=}"
            if args.ntfy:
                ntfy = Ntfy(topic=os.environ.get("NTFY_TOPIC", None))
                ntfy.send_notification(message)
            print(message)

    ############################
    # Run
    ############################
    trainer.fit(model=sae_module, datamodule=dm)


if __name__ == "__main__":
    main()
