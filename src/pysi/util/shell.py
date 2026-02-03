"""Run shell commands.

This sub-module contains functions for running useful things typically done
in a shell environment.
"""

import re
import subprocess
from pathlib import Path


def run_shell_command(
    command: list[str] | str,
    file_path: str | Path = Path(),
    *,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run a shell command.

    Parameters
    ----------
    command : List[str] or str
        The shell command to run. Must either be a single string to call a
        program, or a list of the program and arguments for the program.
    file_path : str or pathlib.Path [optional]
        The directory to run the command in.
    verbose : bool
        Print stdout to the screen.

    Returns
    -------
    shell_out: subprocess.CompletedProcess
        The output of the shell command.

    """
    shell_out = subprocess.run(command, capture_output=True, cwd=file_path, check=True)  # noqa: S603
    if verbose:
        print(shell_out.stdout.decode("utf-8"))  # noqa: T201
    if shell_out.stderr:
        print(  # noqa: T201
            f"Errors were reported for {' '.join(command)}:\n",
            shell_out.stderr.decode("utf-8"),
        )

    return shell_out


def find_file_with_pattern(pattern: str, file_path: Path | str = Path()) -> list[str]:
    """Find files of the given pattern recursively.

    This is used to find a number files given a global pattern, i.e. \*.spec,
    \*.pf. When \*.py is used, it'll ignore out.pf and py_wind.pf files. To find
    py_wind.pf files, use py_wind.pf as the pattern.

    Parameters
    ----------
    pattern : str
        Patterns to search recursively for, i.e. \*.pf, \*.spec, tde_std.pf
    file_path : str [optional]
        The directory to search from, if not specified in the pattern.

    Returns
    -------
    list[str]
        The list of files found.

    """
    file_path = Path(file_path).expanduser()
    files = [str(file_) for file_ in Path(f"{file_path}").rglob(pattern)]
    if ".pf" in pattern:
        files = [this_file for this_file in files if "out.pf" not in this_file and "py_wind" not in this_file]

    try:
        files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])
    except TypeError:
        files.sort()

    return files
