from typing import Any, Dict, Union

import wandb


def get_current_wandb_run_name() -> str:
    """
    Get the name of the current wandb run.
    """

    return wandb.run.name


def convert_to_non_wandb_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the config without the wandb config.
    """

    config = config["parameters"]

    new_config: Dict[str, str] = {}

    for key, value in config.items():
        new_config[key] = value["values"][0]

    return new_config


def create_method_name_for_wandb(method_name: str, wandb_config: Dict[str, Union[str, int, float, bool]]) -> str:
    """
    Create the method name for wandb.
    """

    for key, value in wandb_config.items():

        if key not in ['result_path', 'method_name'] and 'path' not in key:
            if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
                method_name += f"___{key}_{str(value).lower()}"
            elif isinstance(value, list):
                method_name += f"___{key}_{value}"
            elif value is not None:
                beautified_param = value.replace('-', '').replace(' ', '_')
                method_name += f"___{key}_{beautified_param}"

        if len(method_name) > 100:
            break

    return method_name
