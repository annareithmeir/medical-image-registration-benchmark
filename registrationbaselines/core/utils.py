from pathlib import Path
import datetime
import subprocess

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


def run_command_in_terminal(command: list[str], check: callable) -> bool:
    """
        Run a command in the terminal and check the result using the provided lambda function.
    """
    try:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = p.communicate()

        if p.returncode != 0 or not check():

            error_message = ''
            if not check():
                error_message += 'Check failed\n\n'
            if p.returncode != 0:
                error_message += 'Command returned non-zero exit code\n\n'
            error_message += str(output[1])

            raise FileNotFoundError(error_message)

    except OSError as e:
        print(e)
        print('Is the tool correctly installed?')

        return False
    
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
