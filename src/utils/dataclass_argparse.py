"""
dataclass_argparse.py

This module provides an extension to Python's `argparse.ArgumentParser` that allows
users to add dataclass configurations as command-line arguments. With `DataclassArgumentParser`,
arguments from multiple dataclasses can be added to the parser, allowing both positional and
optional arguments. This parser handles deduplication of argument names and can automatically
populate instances of each dataclass with the parsed values from the command line.

Features:
    - Supports field-level help strings through `metadata={'help': "help text"}` in dataclass fields.
    - Deferred addition of arguments, ensuring that positional arguments are added before optional arguments.
    - Ability to parse arguments directly as instances of the specified dataclasses.
    - Allows overriding positional arguments with optional arguments where names may overlap.

Example:
    ```
    @dataclass
    class TrainingConfig:
        model_name: str
        learning_rate: float
        batch_size: int = 32
        num_epochs: int = 10

    parser = DataclassArgumentParser(description="Parser with dataclass support.")
    parser.add_dataclass(TrainingConfig)
    configs = parser.parse_dataclasses()
    training_config = configs[0]
    ```

    This script will support CLI commands such as:
    ```
    python script.py my_model --learning-rate 0.01 --batch-size 64
    ```

Author:
    Austin Davis, 2024
"""

import argparse
import inspect
from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Type, Union


class DataclassArgumentParser(argparse.ArgumentParser):
    """
    An ArgumentParser extension that supports adding dataclass fields as command-line arguments.

    This parser allows users to register dataclasses with `add_dataclass`. Arguments are added
    to the parser in a deferred manner, ensuring that positional arguments appear before optional
    arguments. Argument names are deduplicated, with optional arguments overriding positional
    arguments when a name conflict occurs.

    Special metadata tags influence the behavior of a field:
        shared (bool):      Indicates this argument is a placeholder for an parameter defined in a different
                            dataclass. (WARNING: @dataclass will require a dummy default value.) Only apply 
                            this flag to dataclasses that share a common parameter defined elsewhere; do not
                            include this flag in the dataclass that fully-defines the parameter.
        
        required (bool):    If False, parameters lacking a default value are treated as "optional" parameters
        
        action (str):       If present, the parameter is treated like an argparse action string (e.g., 'store_true')

    Attributes:
        frozen (bool): A flag indicating if arguments have already been added to the parser, 
                        preventing duplicate arguments.

    Methods:
        add_dataclass(dataclass_type: Type): Caches a dataclass type for argument parsing, deferring
                                             the actual addition until parsing.
        parse_args(*args, **kwargs): Overrides `parse_args` to add cached dataclass arguments before parsing.
        parse_dataclasses(*args, **kwargs) -> List[Any]: Parses arguments and returns instances of
                                                         each dataclass, populated with CLI argument values.
    """
    frozen = False

    def __init__(self, dataclasses: Optional[Union[Type, List[Type]]] = None, *args, **kwargs):
        """
        Initializes the DataclassArgumentParser. Optionally accepts a list of dataclass types to
        be added directly to the parser during initialization.

        Args:
            dataclasses (Optional[Union[Type, List[Type]]]): A list of dataclass types to add to the parser.
            *args: Positional arguments for the ArgumentParser superclass.
            **kwargs: Keyword arguments for the ArgumentParser superclass.
        """
        super().__init__(*args, **kwargs)
        self._dataclass_cache = {}

        # Automatically add any dataclasses provided at initialization
        if dataclasses:
            if not isinstance(dataclasses, Iterable):
                dataclasses = [dataclasses]
            for dataclass_type in dataclasses:
                self.add_dataclass(dataclass_type)

    def add_dataclass(self, dataclass_type: Type):
        """
        Caches a dataclass configuration for later processing. The actual
        argument addition is deferred until parsing to ensure positional
        arguments come before optional ones.

        Args:
            dataclass_type (Type): The dataclass type to cache for argument parsing.
        
        Raises:
            TypeError: If `dataclass_type` is not a dataclass.
            ValueError: If arguments have already been parsed.
        """
        if self.frozen:
            raise ValueError("Cannot add a dataclass after parsing args")

        if not is_dataclass(dataclass_type):
            raise TypeError("add_dataclass only accepts dataclass types")

        # Separate fields into positional and optional
        positional_fields = []
        optional_fields = []
        shared_fields = []

        for field in fields(dataclass_type):
            
            if field.metadata.get("shared",False):
                shared_fields.append(field)
                continue # field is defined elsewhere

            if field.metadata.get("action",None):
                assert not field.metadata.get("required", False), "Actions cannot be required fields"

            if (
                not field.metadata.get("action", None) # actions are always optional
                and field.default == MISSING
                and field.default_factory == MISSING
                and field.metadata.get("required", True) # assume required if no default is given
            ):
                positional_fields.append(field)
            else:
                optional_fields.append(field)

        # Cache the dataclass with its fields separated by type
        self._dataclass_cache[dataclass_type.__name__] = (dataclass_type, positional_fields, optional_fields, shared_fields)
        return self

    def _add_dataclass_arguments(self):
        """Adds arguments for each cached dataclass, ensuring positional arguments precede optional ones."""
        if self.frozen:
            return

        fields = {}
        # de-duplicate arguments, allowing defaults to override positionals
        for dataclass_type, positional_fields, optional_fields, shared_fields in self._dataclass_cache.values():
            # do nothing for shared fields; they're defined elsewhere
            
            for field in positional_fields:
                fields[field.name] = (field, True)
            for field in optional_fields:
                fields[field.name] = (field, False)
            

        # Add positional arguments first:
        for field in [f for (f, is_positional) in fields.values() if is_positional]:
            help_text = field.metadata.get('help', self._get_field_help(dataclass_type, field.name))
            self.add_argument(
                field.name,
                type=field.type,
                help=help_text,
            )
        # add optional arguments
        for field in [f for (f, is_positional) in fields.values() if not is_positional]:
            arg_name = f"--{field.name}"
            help_text = field.metadata.get('help', self._get_field_help(dataclass_type, field.name))
            action_str = field.metadata.get('action', None)
            if field.default != MISSING:
                help_text = " ".join([help_text, f"Default= {field.default}"])
            if action_str:
                self.add_argument(
                    arg_name,
                    action=action_str,
                    help=help_text,
                )
            else:
                self.add_argument(
                    arg_name,
                    type=field.type,
                    required=False, # optional areguments can't be required
                    help=help_text,
                    default=field.default if field.default != MISSING else field.default_factory
                )
        self.frozen = True

    def _get_field_help(self, dataclass_type: Type, field_name: str) -> str:
        """Extracts the help string from a field's docstring if available."""
        for line in inspect.getdoc(dataclass_type).splitlines():
            if line.strip().startswith(f"{field_name}:"):
                return line.split(':', 1)[1].strip()
        return ""

    def parse_args(self, *args, **kwargs) -> argparse.Namespace:
        """Override parse_args to process cached dataclass arguments before parsing."""
        self._add_dataclass_arguments()  # Ensure arguments are added just before parsing
        return super().parse_args(*args, **kwargs)

    def parse_dataclasses(self, *args, **kwargs) -> dict[Any]:
        """
        Parses command-line arguments and returns a dictionary of dataclass instances.
        Each instance corresponds to a dataclass whose arguments were added to the parser.

        Returns:
            dict[Any]: A dictionary of dataclass instances with arguments populated from the command-line input.

        Example Usage:
        ```
            data_classes = parser.parse_dataclasses()
            cfg: Config = data_classes[Config]
            opts: Options = data_classes[Options]
        ```
        """
        self._add_dataclass_arguments()  # Ensure arguments are added just before parsing
        args = self.parse_args(*args, **kwargs)
        parsed_args = vars(args)

        instances = {}
        for (dataclass_type, positional_fields, optional_fields, shared_fields) in self._dataclass_cache.values():
            # Gather arguments specific to this dataclass
            dataclass_args: Dict[str, Any] = {
                field.name: parsed_args.get(field.name) or parsed_args.get(field.name.replace('_', '-'))
                for field in (positional_fields + optional_fields + shared_fields)
                if field.name in parsed_args or f"--{field.name.replace('_', '-')}" in parsed_args
            }
            # Instantiate the dataclass with the parsed arguments
            instances[dataclass_type] = (dataclass_type(**dataclass_args))

        instances['parsed_args'] = parsed_args
        return instances

# Example dataclass configuration
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str               # Name of the model (positional argument)
    learning_rate: float   # Learning rate for the optimizer
    batch_size: int = 32          # Size of each training batch
    num_epochs: int = 10          # Number of training epochs

@dataclass
class ModelConfig:
    """Configuration for model architecture.
    model_name: [str] Name of the model 
    hidden_size: [int] Size of the model (Default: 512)
    num_layers: [int] Number of layers
    """
    model_name: str  = "French"              # Name of the model (positional argument)
    hidden_size: int = 512        # Hidden size of the model
    num_layers: int = 6           # Number of layers in the model
    activation: str = 'relu'      # Activation function to use

# Usage example
if __name__ == "__main__":
    parser = DataclassArgumentParser(description="Parser with dataclass support.")
    
    # Add dataclasses to the parser
    parser.add_dataclass(TrainingConfig)
    parser.add_dataclass(ModelConfig)
    parser.parse_dataclasses('--help'.split())
    # Parse as dataclasses
    configs = parser.parse_dataclasses('5 --model_name f12'.split())
    
    # Access each dataclass instance
    training_config, model_config = configs
    print("Training Config:", training_config)
    print("Model Config:", model_config)
