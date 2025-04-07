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
        decoder_LFD: nn.ModuleList,  # assume type: nn.ModuleList[nn.Linear]
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
        model_activations_BLPD: torch.Tensor,
        reconstructed_model_activations_BLPD: torch.Tensor,
        encoded_representations_BF: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_LFD: nn.ModuleList,  # assume type: nn.ModuleList[nn.Linear]
    ):
        recon_loss = F.mse_loss(
            reconstructed_model_activations_BLPD,
            model_activations_BLPD,
            reduction="mean",
            weight=attention_mask,
        )
        return recon_loss


class CrossCoderL1Loss(nn.Module):
    r"""
    L1-of-norms version of the crosscoder loss from:
    https://transformer-circuits.pub/2024/crosscoders/index.html

    The loss uses the L2 loss of the true and reconstructed activations, with
    a L1 regularization term that sums the weights of the intermediate dictionary
    feature vectors scaled by the L1 norm of the weights in the decoder.

    Loss is computed as:

        L = ∑ₗ ‖aˡ(xⱼ) - âˡ(xⱼ)‖² + ∑ᵢ fᵢ(xⱼ) * (∑ₗ ‖Wˡ_{dec, i}‖)

    where:
        - aˡ(xⱼ): true activations for layer l
        - âˡ(xⱼ): reconstructed activations via decoder
        - fᵢ(xⱼ): feature i activations
        - Wˡ_{dec, i}: decoder vector for feature i at layer l
    """

    def __init__(
        self,
        #     decoder_LFD: nn.ModuleList,  # assume type: nn.ModuleList[nn.Linear]
    ):
        #     """
        #     :param decoder_LFD: Dictionary mapping layer index l to decoder module W^l_{dec}
        #     :type decoder_LFD: nn.ModuleList[nn.Linear]
        #     """
        super().__init__()

    #     self.decoder_weights = [linear.weight for linear in decoder_LFD]

    def forward(
        self,
        model_activations_BLPD: torch.Tensor,
        reconstructed_model_activations_BLPD: torch.Tensor,
        encoded_representations_BPF: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_LFD: nn.ModuleList,  # assume type: nn.ModuleList[nn.Linear]
    ) -> torch.Tensor:

        # MSE reconstruction loss
        recon_loss = F.mse_loss(
            reconstructed_model_activations_BLPD,
            model_activations_BLPD,
            reduction="mean",
            weight=attention_mask,
        )

        # Compute regularization term:
        decoder_weights = [linear.weight for linear in decoder_LFD]
        decoder_weights_LFD = torch.stack(decoder_weights)
        decoder_norms_LF = decoder_weights_LFD.norm(dim=1)  # L2 of weights
        decoder_l1_of_norms_F = decoder_norms_LF.sum(dim=0)  # L1 of L2 norms
        reg_loss = encoded_representations_BPF @ decoder_l1_of_norms_F  # scale encoding by L1 norms

        # Total loss: sum over batch
        total_loss = recon_loss + reg_loss.sum()
        return total_loss

        """
        :param model_activations_BLPD: Feature activations f(x_j), shape (batch, F)
        :type feature_activations: Tensor
        :param reconstructed_model_activations_BLPD: Dict mapping layer index l to true activations a^l(x_j), each of shape (batch, D_l)
        :type layer_targets: dict[int, Tensor]
        :param encoded_representations_BPF: 
        :type encoded_representations_BPF: 
        :return: Scalar loss value
        :rtype: Tensor
        """
