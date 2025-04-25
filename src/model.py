from typing import Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sae import SparseAutoEncoder


class SaeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        model_activations_BD: torch.Tensor,
        reconstructed_model_activations_BD: torch.Tensor,
        encoded_representations_BF: torch.Tensor,
    ) -> torch.Tensor: ...


class L1WeightedLoss(SaeLoss):

    def __init__(self, l1_coefficient: float):
        super().__init__()
        self.l1_coefficient = l1_coefficient

    def forward(
        self,
        model_activations_BD: torch.Tensor,
        reconstructed_model_activations_BD: torch.Tensor,
        encoded_representations_BF: torch.Tensor,
    ):
        l2_loss = F.mse_loss(
            reconstructed_model_activations_BD,
            model_activations_BD,
        )

        l1_loss = self.l1_coefficient * encoded_representations_BF.abs().sum()
        loss = l2_loss + l1_loss
        return loss


class MSELoss(SaeLoss):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        model_activations_BD: torch.Tensor,
        reconstructed_model_activations_BD: torch.Tensor,
        encoded_representations_BF: torch.Tensor,
    ):
        return F.mse_loss(
            reconstructed_model_activations_BD,
            model_activations_BD,
        )


class ConstrainedAdam(torch.optim.Adam):
    """
    Lifted from https://github.com/saprmarks/dictionary_learning/blob/07975f7a7c505042b6619846db18dbd122c4f4e6/dictionary_learning/trainers/trainer.py#L38

    A variant of Adam where some of the parameters are constrained to have unit norm.
    Note: This should be used with a decoder that is nn.Linear, not nn.Parameter.
    If nn.Parameter, the dim argument to norm should be 1.
    """

    def __init__(
        self,
        params,
        constrained_params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
    ):
        super().__init__(params, lr=lr, betas=betas)
        self.constrained_params = list(constrained_params)

    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                if p.grad is None:
                    continue
                normed_p = p / p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=0, keepdim=True)


class LightSparseAutoEncoder(L.LightningModule):
    train_loss_fn: nn.Module
    """loss function used during training (e.g. L1-weighted reconstruction loss)."""
    val_loss_fn: nn.Module
    """loss function used during validation (e.g., MSELoss)"""
    lr: float
    """Learning rate used for optimization."""

    def __init__(
        self,
        activation_dim: int,
        dict_size: int = None,
        dict_scale: int = None,
        pre_activation: Literal["relu", "none"] = "relu",
        sparsity_scheme: Literal["topk", "batch_topk"] = "batch_topk",
        sparsity_parameter: int = 32,
        train_loss_fn_str: Literal["mse", "weightedL1"] = "mse",
        val_loss_fn_str: Literal["mse", "weightedL1"] = "mse",
        automatic_optimization: bool = True,
        initial_learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.sae = SparseAutoEncoder(
            activation_dim=activation_dim,
            dict_size=dict_size,
            dict_scale=dict_scale,
            pre_activation=pre_activation,
            sparsity_scheme=sparsity_scheme,
            sparsity_parameter=sparsity_parameter,
        )

        self.automatic_optimization = automatic_optimization
        self.lr = initial_learning_rate

        supported_losses = {"mse": MSELoss(), "weightedL1": L1WeightedLoss(sparsity_parameter)}
        self.train_loss_fn = supported_losses[train_loss_fn_str]
        self.val_loss_fn = supported_losses[val_loss_fn_str]

    def forward(self, model_activations_D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param model_activations_D: Activations tensor with shape [..., d_model]
        :type model_activations_D: Tensor
        :return: Tuple of (reconstructed activations [..., D], encoded representation [..., F]).
        :rtype: tuple[Tensor, Tensor]
        """
        return self.sae.forward(model_activations_D)

    def configure_optimizers(self):
        optimizer = ConstrainedAdam(self.sae.parameters(), self.sae.decoder_FD.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self):
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("lr", lr, prog_bar=True, logger=True)

    def step(self, model_activations_BD: torch.Tensor, loss_fn: SaeLoss):
        reconstructed_model_activations_BD, encoded_representation_BF = self.forward(model_activations_BD)
        loss = loss_fn(
            model_activations_BD,
            reconstructed_model_activations_BD,
            encoded_representation_BF,
        )
        return loss

    def training_step(self, model_activations_BD: torch.Tensor):
        loss = self.step(model_activations_BD, self.train_loss_fn)
        self.log("train/loss", loss)
        return {"loss": loss}

    def validation_step(self, model_activations_BD: torch.Tensor):
        loss = self.step(model_activations_BD, self.val_loss_fn)
        self.log("validation/loss", loss)
        return {"loss": loss}

    def test_step(self, model_activations_BD: torch.Tensor):
        loss = self.step(model_activations_BD, self.val_loss_fn)
        self.log("test/loss", loss)
        return {"loss": loss}
