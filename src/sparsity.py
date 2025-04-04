import torch
import torch.nn as nn


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
        flat_input = input_BX.flatten(end_dim=1)
        _, indices = flat_input.topk(self.k * batch_size, dim=dim)

        mask = torch.zeros_like(flat_input, dtype=torch.bool).scatter(
            dim=dim,
            index=indices,
            value=True,
        )

        return input_BX * mask.view_as(input_BX)
