"""
Implementation of SAE inspired by https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html

"""

import math

import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

from src.losses import CrossCoderL1Loss, MSELoss, SaeLoss


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


class SparseCrosscoder(L.LightningModule):
    """A Sparse Crosscoder (SCC) module for learning sparse representations of transformer residual stream activations.

    Encodes activations into a higher-dimensional sparse latent space using a learned feature dictionary,
    then decodes them back to reconstruct the original activations. The linear layers in the encoder share a single bias
    term, whereas the linear layers of the decoder each have their own bias terms.
    """

    activation_dim: int
    """(D) Size of the input activation vectors"""
    dict_size: int
    """(F) Size of the SAE's feature dictionary, i.e. the expanded latent space"""
    encoder_LDF: nn.ModuleList  # type: nn.ModuleList[nn.Linear]
    """List of (L) linear encoders which map from d_model (D) to feature dictionary (F)"""
    encoder_bias_F: nn.Parameter
    """The single, shared bias term common to all layers of the encoder"""
    decoder_LFD: nn.ModuleList  # type: nn.ModuleList[nn.Linear]
    """List of (L) linear decoders which map from feature dictionary (F) to d_model (D). Each layer has its own bias."""
    activation_fn: nn.Module
    """Nonlinear activation applied after both encoding and decoding."""
    loss_fn: nn.Module
    """oss function used during training (e.g. L1-weighted reconstruction loss)."""
    lr: float = 0.0001
    """Learning rate used for optimization."""

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        n_layers: int,
        activation_fn: nn.Module = nn.ReLU(),
        automatic_optimization: bool = True,
    ):
        super().__init__()

        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.automatic_optimization = automatic_optimization

        # initialize encoder
        self.encoder_LDF = nn.ModuleList(
            [nn.Linear(activation_dim, dict_size, bias=False) for _ in range(n_layers)]
        )
        # initialize bias
        self.encoder_bias_F = nn.Parameter(torch.empty(dict_size))
        bound = 1 / math.sqrt(self.activation_dim) if self.activation_dim > 0 else 0
        init.uniform_(self.encoder_bias_F, -bound, bound)

        # initialize decoder weights
        self.decoder_LFD = nn.ModuleList(
            [nn.Linear(dict_size, activation_dim, bias=True) for _ in range(n_layers)]
        )
        self.activation_fn = activation_fn
        self.train(True)

        self.save_hyperparameters(ignore=["loss_fn", "activation_fn"])

    def train(self, mode=True):
        if mode:
            self.loss_fn = CrossCoderL1Loss()
        else:
            self.loss_fn = MSELoss()
        return super().train(mode)

    def encode(self, model_activations_BLPD: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input activation vectors into a sparse latent representation.

        :param model_activations_LD: Input activations of shape [B, L, P, D].
        :type model_activations_LD: Tensor
        :return: Encoded sparse representation of shape [B, P, F].
        :rtype: Tensor"""

        outputs = [
            linear(model_activations_BLPD[:, layer_idx])
            for layer_idx, linear in enumerate(self.encoder_LDF)
        ]

        summed = torch.stack(outputs).sum(dim=0)

        return self.activation_fn(summed + self.encoder_bias_F)

    def decode(self, encoded_representation_BPF: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back into the original activation space.

        :param encoded_representation_BPF: Sparse representation of shape [B, P, F].
        :type encoded_representation_BPF: Tensor
        :return: Reconstructed activations of shape [B, L, P, D].
        :rtype: Tensor"""

        outputs = [linear(encoded_representation_BPF) for linear in self.decoder_LFD]

        return torch.stack(outputs, dim=1)  # dim=1 puts batch at index 0 as desired

    def forward(
        self,
        model_activations_BLPD: torch.Tensor,
        attention_mask_BLPD: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass on a batch of the model activations. If an attention mask is given,
        the loss will be computed on the batched inputs
        :param model_activations_BLPD: Input activations of shape [Batch, Layers, Position, D_model].
        :type model_activations_BLPD: torch.Tensor
        :param attention_mask_BLPD: Attention mask used if input activations were padded.
        :type attention_mask_BLPD: torch.Tensor

        """
        encoded_representation_BPF = self.encode(model_activations_BLPD)
        reconstructed_model_activations_BLPD = self.decode(encoded_representation_BPF)

        if attention_mask_BLPD is not None:
            loss = self.loss_fn(
                model_activations_BLPD,
                reconstructed_model_activations_BLPD,
                encoded_representation_BPF,
                attention_mask_BLPD,
                self.decoder_LFD,
            )
            return reconstructed_model_activations_BLPD, encoded_representation_BPF, loss
        else:
            return reconstructed_model_activations_BLPD, encoded_representation_BPF

    def configure_optimizers(self):
        optimizer = ConstrainedAdam(self.parameters(), self.decoder_LFD.parameters(), lr=self.lr)
        return {"optimizer": optimizer}

    def step(self, model_activations_BLPD: torch.Tensor, attention_mask_BLPD: torch.Tensor = None):
        *_, loss = self.forward(model_activations_BLPD, attention_mask_BLPD)

        return loss

    def training_step(self, batch_data: tuple[torch.Tensor, torch.Tensor]):
        model_activations_BLPD, attention_mask_BLPD = batch_data
        loss = self.step(model_activations_BLPD, attention_mask_BLPD)
        self.log("train/loss", loss)
        return {"loss": loss}

    def validation_step(self, batch_data: tuple[torch.Tensor, torch.Tensor]):
        model_activations_BLPD, attention_mask_BLPD = batch_data
        loss = self.step(model_activations_BLPD, attention_mask_BLPD)
        self.log("validation/loss", loss)
        return {"loss": loss}

    def test_step(self, batch_data: tuple[torch.Tensor, torch.Tensor]):
        model_activations_BLPD, attention_mask_BLPD = batch_data
        loss = self.step(model_activations_BLPD, attention_mask_BLPD)
        self.log("test/loss", loss)
        return {"loss": loss}
