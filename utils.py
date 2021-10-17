import os

from colorama import Fore, Style


def yes_or_no(question):
    """Creates y/n question with handling invalid inputs within console

    :param question: required question string
    :type question: String
    :return: True or False for the input
    :rtype: bool
    """
    while "the answer is invalid":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if reply[0] == "y":
            return True
        if reply[0] == "n":
            return False


def print_results(total_cnt, errors, skip_cnt, print_failed=True):
    error_cnt = len(errors)
    success_cnt = total_cnt - error_cnt - skip_cnt
    print(
        f"{success_cnt} items extracted, {skip_cnt} items skipped, {error_cnt} items failed"
    )
    if print_failed and errors:
        print("Failed items:")
        for item in errors:
            print(item)


def get_paths(in_path):
    """
    returns list of the absolute paths of the files in the input dir

    :param in_path: parent path of the files
    :type in_path: String or PathLike object
    :return: list of absolute paths of files
    :rtype: list
    """
    # not including subdirectory files
    path = os.path.abspath(in_path)
    file_list = [
        x.path
        for x in os.scandir(path)
        if (x.path.endswith(".wav") or x.path.endswith(".WAV"))
    ]
    return file_list


def get_filename(in_path):
    """
    returns only file name without extension

    :param in_path: path to get filename
    :type in_path: String or PathLike object
    :rtype: os.path object
    """
    return os.path.splitext(os.path.basename(in_path))[0]


def is_dir(in_path):
    if not os.path.isdir(in_path):
        print(Fore.RED + f"Path is not exist: {in_path}" + Style.RESET_ALL)
        if not yes_or_no("Would you like to make one?"):
            return False
        else:
            os.mkdir(in_path)
    return True
