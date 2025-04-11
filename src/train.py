import os

import hydra
import pyrootutils
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig

from src.utils.ntfy import Ntfy

torch.set_float32_matmul_precision("medium")

_root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".env"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": (
        os.path.join(_root, os.environ.get("HYDRA_CONFIG_PATH", None))
        if os.environ.get("HYDRA_CONFIG_PATH", None)
        else str(_root / "configs")
    ),
    "config_name": "train.yaml",
}


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig):

    # get number of cpus allocated to the job
    allocated_cpus = len(os.sched_getaffinity(0))

    data = instantiate(
        cfg.data.instance,
        num_workers=min(cfg.data.max_workers, allocated_cpus),  # 32 is max recommended by pt lightning
        num_proc=min(32, allocated_cpus),  # use as many as allocated to load Hf dataset
    )

    model = instantiate(cfg.model)

    callbacks: list[Callback] = instantiate(cfg.callbacks)

    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks)

    if "tuner" in cfg:
        tuner = Tuner(trainer=trainer)
        results = {}
        if "lr_find" in cfg.tuner:
            initial_lr = tuner.lr_find(model=model, datamodule=data)

            results["initial_lr"] = initial_lr.results
        if "scale_batch_size" in cfg.tuner:
            scale_batch_size = tuner.scale_batch_size(model=model, datamodule=data)
            results["scale_batch_size"] = scale_batch_size

        ntfy = Ntfy(topic=os.environ.get("NTFY_TOPIC", None))
        ntfy.send_notification(f"Tuner Finished: {list(results.items())}")

    trainer.fit(model=model, datamodule=data)

    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            filename = cb.last_model_path.replace(".ckpt", "-statedict.pt")
            torch.save(model.state_dict(), filename)
            print(f"Saved state dict to: {filename}")

    trainer.test(model=model, datamodule=data)


if __name__ == "__main__":
    main()
