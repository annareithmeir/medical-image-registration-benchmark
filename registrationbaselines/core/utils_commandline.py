from pathlib import Path
import datetime
import subprocess
import inspect

from typing import Tuple


def create_result_paths(directory: Path,
                        name_fixed: str,
                        name_moving: str,
                        method: str,
                        extension_image: str = ".nii",
                        extension_transformatoin: str = ".tfm") -> Tuple[Path, Path]:
    
    """
    Create the paths for the result files (warped image and transformatoin)."""
    
    date_time = datetime.datetime.now()

    result_transformed_image_path = directory / f"{name_moving}_warped_on_{name_fixed}_{method}_image_{date_time}"
    result_transformation_path = directory / f"{name_moving}_warped_on_{name_fixed}_{method}_transform_{date_time}"

    # replace spaces with underscores
    result_transformed_image_path = result_transformed_image_path.resolve().as_posix().replace(" ", "_")
    result_transformation_path = result_transformation_path.resolve().as_posix().replace(" ", "_")

    # replace - with underscores
    result_transformed_image_path = result_transformed_image_path.replace("-", "_")
    result_transformation_path = result_transformation_path.replace("-", "_")

    # replace : with underscores
    result_transformed_image_path = result_transformed_image_path.replace(":", "_")
    result_transformation_path = result_transformation_path.replace(":", "_")

    # replace . with underscores
    result_transformed_image_path = result_transformed_image_path.replace(".", "_")
    result_transformation_path = result_transformation_path.replace(".", "_")

    # add the extension
    result_transformed_image_path += extension_image
    result_transformation_path += extension_transformatoin

    return Path(result_transformed_image_path), Path(result_transformation_path)


def run_command_in_terminal(command: list[str], check: callable = None) -> bool:
    """
        Run a command in the terminal and check the result using the provided lambda function.
    """
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.communicate()

    if check is not None:
        if not check():
            error_message = f'Check function: {check.__name__} failed\n\n'
            raise FileNotFoundError(error_message + ".\n" + str(output))
    
    return True


def print_command(cmd_list):
    """
        Print the command line as a string, so that it can be copied and pasted into the terminal. For debugging.
    """
    print('\n\n')

    full_cmd = ""

    for a in cmd_list:
        full_cmd += str(a) + " "

    print(full_cmd)


def add_configuration_to_command(command: list[str], configuration: dict):
    """
    Add the configuration to the command line.
    """
    for key, value in configuration.items():
        if isinstance(value, bool):
            if value:  # Only add flag if True
                command.append(f"-{key}")
        else:
            command.append(f"-{key}")
            command.append(f"{value}")

    return command
