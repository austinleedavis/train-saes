import os

import dotenv
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner

from src.callbacks import NtfyCallback, WandbLogger
from src.data import SaeDataModule, SingleLayerHiddenStateCollator
from src.losses import MSELoss
from src.model import SparseAutoEncoder
from src.sparsity import BatchTopKFilter, TopKFilter

torch.set_float32_matmul_precision("medium")


def main():
    dotenv.load_dotenv()

    dm = SaeDataModule(
        data_root=os.environ.get("DATA_ROOT", "data"),
        collator=SingleLayerHiddenStateCollator(layer=10),
        batch_size=1,
        num_workers=15,
    )

    sae = SparseAutoEncoder(
        activation_dim=768,
        dict_size=768 * 4,
        loss_fn=MSELoss(),
        activation_fn=TopKFilter(k=16),
    )

    trainer = Trainer(
        accelerator="auto",
        fast_dev_run=False,
        limit_test_batches=0,
        # limit_val_batches=0,
        max_epochs=10,
        val_check_interval=1.0,  # When using an IterableDataset you must set the val_check_interval to 1.0
        callbacks=[
            NtfyCallback(os.environ.get("NTFY_TOPIC", None)),
            ModelCheckpoint(
                dirpath="models",
                monitor="val_loss",  # metric to monitor
                mode="min",  # minimize val_acc
                save_top_k=2,  # keep only top 2 checkpoints
                save_last=True,  # also save the last checkpoint
                every_n_train_steps=10_000,  # checkpoint every 1000 training steps
                filename="epoch-{epoch}-step-{step}-{val_loss:.4f}",  # custom filename
            ),
        ],
        logger=WandbLogger(
            name="test",
            project="TrainSae",
            log_model=False,
            checkpoint_name=None,
        ),
    )

    trainer.fit(model=sae, datamodule=dm)


if __name__ == "__main__":
    main()
