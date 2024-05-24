from pathlib import Path
import yaml


def read_config(file_path: Path):
    """
    Read the configuration file.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)




