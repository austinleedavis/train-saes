import os
from typing import Union

import torch

LOCAL_PATH_LIKE = Union[str, os.PathLike]


class ModelPersistenceMixin:
    """
    Mixin class providing functionality to save and load PyTorch models.

    This mixin assumes that the class using it has:
    - A `config` attribute (to store hyperparameters).
    - A `state_dict()` method (inherited from `torch.nn.Module`) to save model weights.
    """

    def save_pretrained(self, save_directory: LOCAL_PATH_LIKE):
        """
        Saves the model configuration and state dictionary to the specified directory.

        Args:
            save_directory (str or os.PathLike): Path to the directory where the model should be saved.
        """
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.config, os.path.join(save_directory, "config.pt"))
        torch.save(self.state_dict(), os.path.join(save_directory, "state_dict.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: LOCAL_PATH_LIKE):
        """
        Loads a pretrained model from the specified directory.

        Args:
            load_directory (str or os.PathLike): Path to the directory containing the saved model.

        Returns:
            An instance of the model class with loaded weights and configuration.
        """
        config = torch.load(os.path.join(load_directory, "config.pt"))
        state_dict = torch.load(os.path.join(load_directory, "state_dict.pt"))
        loaded_model = cls(config)
        loaded_model.load_state_dict(state_dict, strict=True)
        return loaded_model
