"""
Implementation of SAE inspired by https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html

"""

import lightning as L
import torch
import torch.nn as nn

from src.losses import L1WeightedLoss, SaeLoss


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


class SparseAutoEncoder(L.LightningModule):
    """A Sparse Autoencoder (SAE) module for learning sparse representations of transformer residual stream activations.

    Encodes activations into a higher-dimensional sparse latent space using a learned feature dictionary,
    then decodes them back to reconstruct the original activations. Typically trained with a sparsity-inducing loss.
    """

    activation_dim: int
    """(D) Size of the input activation vectors"""
    dict_size: int
    """(F) Size of the SAE's feature dictionary, i.e. the expanded latent space"""
    encoder_DF: nn.Linear
    """Linear encoder from d_model (D) to feature dictionary (F)"""
    decoder_FD: nn.Linear
    """Linear decode from feature dictionary (F) to d_model (D)"""
    activation_fn: nn.Module
    """Nonlinear activation applied after both encoding and decoding."""
    loss_fn: nn.Module
    """oss function used during training (e.g. L1-weighted reconstruction loss)."""
    lr: float = 0.001
    """Learning rate used for optimization."""

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        activation_fn: nn.Module = nn.ReLU(),
        loss_fn: SaeLoss = L1WeightedLoss(4),
        automatic_optimization: bool = True,
    ):
        super().__init__()

        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.automatic_optimization = automatic_optimization

        self.encoder_DF = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder_FD = nn.Linear(dict_size, activation_dim, bias=True)
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=["activation_fn", "loss_fn"])

    def encode(self, model_activations_D: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input activation vectors into a sparse latent representation.

        :param model_activations_D: Input activations of shape [..., D].
        :type model_activations_D: Tensor
        :return: Encoded sparse representation of shape [..., F].
        :rtype: Tensor"""
        return self.activation_fn(self.encoder_DF(model_activations_D))

    def decode(self, encoded_representation_F: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back into the original activation space.

        :param encoded_representation_F: Sparse representation of shape [..., F].
        :type encoded_representation_F: Tensor
        :return: Reconstructed activations of shape [..., D].
        :rtype: Tensor"""
        return self.activation_fn(self.decoder_FD(encoded_representation_F))

    def forward(
        self, model_activations_D: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param model_activations_D: Activations tensor with shape [..., d_model]
        :type model_activations_D: Tensor
        :return: Tuple of (reconstructed activations [..., D], encoded representation [..., F]).
        :rtype: tuple[Tensor, Tensor]
        """
        encoded_representation_F = self.encode(model_activations_D)
        reconstructed_model_activations_D = self.decode(encoded_representation_F)
        return reconstructed_model_activations_D, encoded_representation_F

    def configure_optimizers(self):
        optimizer = ConstrainedAdam(
            self.parameters(), self.decoder_FD.parameters(), lr=self.lr
        )
        return {"optimizer": optimizer}

    def step(self, model_activations_BD: torch.Tensor):
        reconstructed_model_activations_BD, encoded_representation_BF = self.forward(
            model_activations_BD
        )
        loss = self.loss_fn(
            model_activations_BD,
            reconstructed_model_activations_BD,
            encoded_representation_BF,
        )
        return loss

    def training_step(self, model_activations_BD: torch.Tensor):
        loss = self.step(model_activations_BD)
        self.log("train/loss", loss)
        return {"loss": loss}

    def validation_step(self, model_activations_BD: torch.Tensor):
        loss = self.step(model_activations_BD)
        self.log("validation/loss", loss)
        return {"loss": loss}

    def test_step(self, model_activations_BD: torch.Tensor):
        loss = self.step(model_activations_BD)
        self.log("test/loss", loss)
        return {"loss": loss}
