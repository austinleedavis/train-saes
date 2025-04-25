"""
Implementation of SAE inspired by https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html
"""

import os
from dataclasses import asdict, dataclass
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file, save_file


class TopKFilter(nn.Module):
    """
    Applies a standard top-k filter to the input tensor by retaining only the top-k activations
    per example along the specified dimension and zeroing out the rest.

    This layer enforces activation sparsity by selecting the top-k highest values for each
    individual sample in the batch independently. It is used in sparse autoencoders (SAEs)
    and other models that benefit from promoting competition among features within a single example.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, input: torch.Tensor, dim=-1):
        _, indices = input.topk(self.k, dim=dim)
        mask = torch.zeros_like(input, dtype=torch.bool).scatter(
            dim=dim,
            index=indices,
            value=True,
        )
        return input * mask

    def extra_repr(self):
        return f"k={self.k}"


class BatchTopKFilter(nn.Module):
    """
    Applies a batched top-k filter to the input tensor by retaining only the top-k activations
    across the entire batch and zeroing out the rest.

    This method implements the BatchTopK activation sparsity strategy described in
    "BatchTopK: A Simple Improvement for TopK SAEs" by Nora Belrose. Unlike standard per-example
    top-k sparsity, which keeps the top-k values for each example independently, BatchTopK selects
    the top-k * B highest activations across the entire batch (where B is the batch size),
    promoting more efficient and global competition among features.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, input_BX: torch.Tensor, dim=-1):
        batch_size = input_BX.shape[0]
        flat_input = input_BX.flatten()
        _, indices = flat_input.topk(self.k * batch_size, dim=dim)

        mask = (
            torch.zeros_like(flat_input, dtype=torch.bool)
            .scatter(
                dim=dim,
                index=indices,
                value=True,
            )
            .reshape_as(input_BX)
        )

        return input_BX * mask

    def extra_repr(self):
        return f"k={self.k}"


@dataclass
class SaeConfig:
    activation_dim: int
    dict_size: int
    pre_activation: str
    sparsity_scheme: str
    sparsity_parameter: int | float

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        file_path = os.path.join(pretrained_model_name_or_path, "config.yaml")
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        config = cls(**data)
        return config

    def save_pretrained(self, pretrained_model_name_or_path: str):
        file_path = os.path.join(pretrained_model_name_or_path, "config.yaml")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.safe_dump(asdict(self), f)


class SparseAutoEncoder(nn.Module):
    """A Sparse Autoencoder (SAE) module for learning sparse representations of transformer residual stream activations.

    Encodes activations into a higher-dimensional sparse latent space using a learned feature dictionary,
    then decodes them back to reconstruct the original activations.
    """

    activation_dim: int
    """(D) Size of the input activation vectors"""
    dict_size: int
    """(F) Size of the SAE's feature dictionary, i.e. the expanded latent space"""
    encoder_DF: nn.Linear
    """Linear encoder from d_model (D) to feature dictionary (F)"""
    decoder_FD: nn.Linear
    """Linear decode from feature dictionary (F) to d_model (D)"""

    def __init__(
        self,
        activation_dim: int,
        dict_scale: int = None,
        dict_size: int = None,
        pre_activation: Literal["relu", "none"] = "relu",
        sparsity_scheme: Literal["topk", "batch_topk"] = "batch_topk",
        sparsity_parameter: int | float = 32,
    ):
        super().__init__()
        if dict_size is not None:
            self.dict_size = dict_size
        elif dict_scale is not None:
            self.dict_size = dict_scale * activation_dim
        else:
            raise ValueError("Must provide either dict_size or dict_scale. Both were None.")

        self.activation_dim = activation_dim

        SUPPORTED_PREACTIVATIONS = {
            "relu": nn.ReLU(),
            "none": nn.Identity(),
        }
        pre_activation_obj = SUPPORTED_PREACTIVATIONS[pre_activation]

        SUPPORTED_SPARISTY_ACTIVATIONS = {
            "topk": TopKFilter(sparsity_parameter),
            "batch_topk": BatchTopKFilter(sparsity_parameter),
            "none": nn.Identity(),
        }
        sparsity_fn = SUPPORTED_SPARISTY_ACTIVATIONS[sparsity_scheme]

        self.encoder_DF = nn.Sequential(
            pre_activation_obj,
            nn.Linear(self.activation_dim, self.dict_size, bias=True),
            sparsity_fn,
        )

        self.decoder_FD = nn.Linear(self.dict_size, self.activation_dim, bias=True)

        self.config = SaeConfig(
            activation_dim=activation_dim,
            dict_size=self.dict_size,
            pre_activation=pre_activation,
            sparsity_scheme=sparsity_scheme,
            sparsity_parameter=sparsity_parameter,
        )

    def encode(self, model_activations_D: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input activation vectors into a sparse latent representation.

        :param model_activations_D: Input activations of shape [batch, D].
        :type model_activations_D: Tensor
        :return: Encoded sparse representation of shape [batch, F].
        :rtype: Tensor"""
        return self.encoder_DF(model_activations_D)

    def decode(self, encoded_representation_F: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back into the original activation space.

        :param encoded_representation_F: Sparse representation of shape [batch, F].
        :type encoded_representation_F: Tensor
        :return: Reconstructed activations of shape [batch, D].
        :rtype: Tensor"""
        return self.decoder_FD(encoded_representation_F)

    def forward(self, model_activations_D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param model_activations_D: Activations tensor with shape [batch, d_model]
        :type model_activations_D: Tensor
        :return: Tuple of (reconstructed activations [batch, D], encoded representation [batch, F]).
        :rtype: tuple[Tensor, Tensor]
        """
        encoded_representation_F = self.encode(model_activations_D)
        reconstructed_model_activations_D = self.decode(encoded_representation_F)
        return reconstructed_model_activations_D, encoded_representation_F

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        config = SaeConfig.from_pretrained(pretrained_model_name_or_path)
        with torch.device("meta"):
            model = cls(asdict(config))

        # Load the weights from the safetensors file
        safetensors_path = f"{pretrained_model_name_or_path}/model.safetensors"
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, assign=True)
        return model

    def save_pretrained(self, pretrained_model_name_or_path):
        self.config.save_pretrained(pretrained_model_name_or_path)
        safetensors_path = f"{pretrained_model_name_or_path}/model.safetensors"
        save_file(self.state_dict(), safetensors_path)
