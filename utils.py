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


def print_results(total_cnt, failed, skip_cnt, print_failed=True):
    failed_cnt = len(failed)
    success_cnt = total_cnt - failed_cnt - skip_cnt
    print(
        f"{success_cnt} items extracted, {skip_cnt} items skipped, {failed_cnt} items failed"
    )
    if print_failed and failed:
        print("Failed items:")
        for item in failed:
            print(item)
