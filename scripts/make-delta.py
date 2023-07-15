import re
import os
import sys
import shutil
import termcolor

from hashlib import md5
from typing import Optional

def print_yellow(msg: str) -> None:
    """
    Prints a string in yellow.
    """
    print(termcolor.colored(msg, "yellow"))

def print_green(msg: str) -> None:
    """
    Prints a string in green.
    """
    print(termcolor.colored(msg, "green"))

def files_differ(file_1: str, file_2: str) -> bool:
    """
    Returns true if files are different.
    """
    return md5(open(file_1, "rb").read()).hexdigest() != md5(open(file_2, "rb").read()).hexdigest()

def get_version_from_directory(directory: str) -> str:
    """
    Gets the version string from a directory.
    """
    matched = re.match(r".*(\d+\.\d+\.\d+).*", directory)
    if not matched:
        raise ValueError(f"Cannot get version from directory {directory}")
    return matched[1]

def delta(version_1_dir: str, version_2_dir: str, output_dir: str) -> None:
    """
    Recursively delta directories/files
    """
    for filename in os.listdir(version_2_dir):
        version_1_file = os.path.join(version_1_dir, filename)
        version_2_file = os.path.join(version_2_dir, filename)
        output_file = os.path.join(output_dir, filename)
        
        if os.path.isdir(version_1_file) and os.path.isdir(version_2_file):
            if not os.path.exists(output_file):
                os.makedirs(output_file)
            delta(version_1_file, version_2_file, output_file)
        elif os.path.isfile(version_1_file) and os.path.isfile(version_2_file):
            if files_differ(version_1_file, version_2_file):
                print_green(f"Copying different file {version_2_file}")
                shutil.copy(version_2_file, output_file)
            else:
                print_yellow(f"Skipping unchanged file {version_2_file}")
        elif os.path.isdir(version_2_file):
            print_green(f"Copying new directory {version_2_file}")
            shutil.copytree(version_2_file, output_file)
        else:
            print_green(f"Copying new file {version_2_file}")
            shutil.copy(version_2_file, output_file)

def main(version_1_dir: str, version_2_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Reads through two directories and makes a delta that, when overwriting version 1,
    will provide the same files as version 2. This does not remove files from version 1
    that are not in version 2.
    """
    if output_dir is None:
        version_1 = get_version_from_directory(version_1_dir)
        version_2 = get_version_from_directory(version_2_dir)
        prefix, _, suffix = os.path.basename(version_1_dir).partition(version_1)
        output_dir = os.path.join(os.path.dirname(version_1_dir), f"{prefix}upgrade-{version_1}-{version_2}{suffix}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Generating delta from {version_1_dir} to {version_2_dir}. Will write to {output_dir}.")
    delta(version_1_dir, version_2_dir, output_dir)

if __name__ == "__main__":
    main(*sys.argv[1:4])
