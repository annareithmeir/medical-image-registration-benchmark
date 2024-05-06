import subprocess


def run_command_in_terminal(command: list[str],
                            check: callable = None,
                            print_command_list: bool = False) -> bool:
    """
        Run a command in the terminal and check the result using the provided lambda function.
    """

    if print_command_list:
        print_command(command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
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
        if key != 'result_path':  # Skip if key is 'result_path'
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    command.append(f"-{key}")
            else:
                command.append(f"-{key}")
                command.append(f"{value}")

    return command
