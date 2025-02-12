import subprocess

from typing import Callable, Optional, List, Any


def run_command_in_terminal(command: list[str],
                            check: Optional[Callable[[], bool]],
                            print_command_list: bool = False) -> bool:
    """
        Run a command in the terminal and check the result using the provided lambda function.
    """

    if print_command_list:
        print_command(command)

    # remove empty command values
    command = [c for c in command if c != '']

    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output = p.communicate()

    if check is not None:
        if not check():
            error_message = f'Check function: {check.__name__} failed\n\n'
            raise FileNotFoundError(error_message + ".\n" + str(output))

    return True


def print_command(cmd_list: List[str]) -> None:
    """
        Print the command line as a string, so that it can be copied and pasted into the terminal. For debugging.
    """
    print('\n\n')

    full_cmd = ""

    for a in cmd_list:
        full_cmd += str(a) + " "

    print(full_cmd)


def add_configuration_to_command(command: List[str],
                                 configuration: Any,
                                 only_value: bool) -> List[str]:
    """
    Add the configuration to the command line.

    @param command: command line arguments
    @param configuration: configuration dictionary (from wandb)

    @return: command line arguments with configuration
    """

    for key, value in configuration.items():

        if key not in ['result_path', 'method_name', 'gpu']:
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    command.append(f"{key}")
            else:

                command.extend(value.split(' '))
                if not only_value:
                    command.append(f"{key}")

    return command
