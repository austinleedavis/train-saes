print("Script started")
import os

import dotenv
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner

from src.callbacks import NtfyCallback, WandbLogger
from src.data import HiddenStateCollator, SaeDataModule
from src.model import SparseCrosscoder
from src.utils.ntfy import Ntfy

torch.set_float32_matmul_precision("medium")


def main():
    print("Reached main()")
    dotenv.load_dotenv()
    data_root = os.environ.get("DATA_ROOT", "data")

    data = SaeDataModule(
        data_root=data_root,
        collator=HiddenStateCollator(),
        batch_size=128,
        num_workers=15,
        num_proc=64,
    )

    model = SparseCrosscoder(
        activation_dim=768,
        dict_size=768 * 4,
        n_layers=12,
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

    # tuner = Tuner(trainer)
    # initial_lr = tuner.lr_find(model=model, datamodule=data)
    # batch_size = tuner.scale_batch_size(model=model, datamodule=data, max_trials=10)

    # ntfy = Ntfy(topic=os.environ.get("NTFY_TOPIC", None))
    # ntfy.send_notification(f"Tuner Finished. {initial_lr.suggestion()=} {batch_size=}")

    trainer.fit(model=model, datamodule=data)


if __name__ == "__main__":
    main()
