from typing import Any, Dict
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
