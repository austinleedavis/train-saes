import torch
import torch.nn as nn
import torch.nn.functional as F


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

        l1_loss = self.l1_coefficient * encoded_representations_BF.sum()
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
