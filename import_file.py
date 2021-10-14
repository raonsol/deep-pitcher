import contextlib
from multiprocessing import freeze_support
from tqdm import tqdm
from line_profiler_pycharm import profile
from colorama import init, Fore, Back, Style
import os
import MySQLdb
import itdbloader
from urllib.parse import unquote
from concurrent import futures
from audio_util import AudioConverter
from itertools import repeat

# 경로와 포맷 지정
ROOT_PATH = os.getcwd()
WAV_PATH = "/Volumes/vault0/dataset3/wav"
CHORUS_PATH = "/Volumes/vault0/dataset3/chorus"
WAV_OVERWRITE = False
CHORUS_OVERWRITE = False


def connect_db():
    print("Loading configuration...", end=" ")
    try:
        config = itdbloader.get_config()
        print(Fore.GREEN + "OK")

    except Exception as e:
        print("Please check if .itdb.config exists")
        return

    print("Checking MySQL connections...", end=" ")
    try:
        cnx = itdbloader.db_connect(config)
        print(Fore.GREEN + "OK")
    except MySQLdb.Error as e:
        print(Fore.RED + "failed")
        print(e)
        print("Please check if MySQL is running, or check the configuration file")
        return
    c = cnx.cursor()
    db_name = config.get("client", "database")

    return c, db_name


def init_db(c, db_name):
    """Create itdb database

    requires: root username and pw in mysql_default.json
    :param c: MySQLdb cursor
    :type c: MySQLdb cursor
    :param db_name: name of the database
    :type db_name: String
    :return: True if database created successfully
    :rtype: bool
    """
    print("Creating database...")
    result = c.execute(f"SHOW DATABASES LIKE '{db_name}';")
    # if db exists
    if result == 1:
        ans = yes_or_no("DB already exists. Do you want to overwrite?")
        if ans is True:
            ans2 = yes_or_no("Are you sure you want to overwrite?")
            if ans2 is True:
                c.execute(f"DROP DATABASE {db_name};")
            else:
                return
        else:
            return

    # create database
    try:
        c.execute(f"CREATE DATABASE {db_name}")
    except MySQLdb.Error as e:
        print(f"{Fore.RED} Failed to create database: {e} {Fore.RESET}")
        return
    print(f"Database created: {db_name}")
    return True


def create_table(c, db_name):
    """Create tracks and playlists table in database

    :param c: MySQLdb cursor
    :type c: MySQLdb cursor
    :param db_name: name of the database
    :type db_name: String
    :return: True if successful
    :rtype: bool
    """
    # create table
    print("Creating table...", end=" ")
    c.execute(f"USE {db_name};")
    try:
        query = open("./itdb.sql").read()
        c.execute(query)
        print(Fore.GREEN + "OK")
    except MySQLdb.Error as e:
        print(Fore.RED + "failed")
        print(e)
        return False
    return True


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


def export_track(data):
    """Export the audio track into WAV file

    :param data: tuple that contains (dict of track info, export path)
    :type data: tuple
    :return: True if export was successful
    :rtype: bool
    """
    track, path = data
    track_path = unquote(track["Location"].replace("file://", ""))
    file_name = str(track["Track ID"]) + ".wav"
    export_path = os.path.join(path, file_name)

    # if the converted file exists, and the overwrite option is False
    if (not WAV_OVERWRITE) and os.path.exists(export_path):
        return -1

    try:
        c = AudioConverter(track_path, export_path)
        c.export(format="wav", sample_rate=16000)
        return 1
    except:
        return track_path


def convert_wav(tracks, out_path):
    # TODO: create option to save full wav file or not

    if not os.path.isdir(out_path):
        print(Fore.RED + "WAV export path is not exist.", end="")
        if yes_or_no("Would you like to make one?"):
            os.mkdir(WAV_PATH)

    error_list = []
    skip_cnt = 0
    target_cnt = len(tracks.values())
    print()
    print(f"Start converting {target_cnt} files...")
    with futures.ProcessPoolExecutor(max_workers=4) as exe:
        results = tqdm(
            exe.map(export_track, zip(tracks.values(), repeat(out_path))),
            total=target_cnt,
        )
        for result in results:
            if result == -1:
                skip_cnt += 1
            elif result != 1:
                error_list.append(result)

    return target_cnt, error_list, skip_cnt


@profile
def _extract_chorus(data):
    track, in_path, out_path = data
    input_path = os.path.join(in_path, track)
    export_path = os.path.join(out_path, track)

    # skip duplicate if option is on
    if not CHORUS_OVERWRITE and os.path.exists(export_path):
        return -1
    try:
        c = AudioConverter(input_path, export_path)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            chorus_sec = c.detect_chorus()
        c.cut_audio(chorus_sec, 30)
        c.export(format="wav", sample_rate=16000)
        return True
    except:
        return in_path


@profile
def extract_chorus(in_path, out_path):
    if not os.path.isdir(out_path):
        print(Fore.RED + "chorus extract path is not exist. ", end="")
        if yes_or_no("Would you like to make one?"):
            os.mkdir(out_path)
        else:
            print("Using default path...")
            out_path = "/tmp"

    # not including subdirectory files
    tracks = [
        x for x in os.listdir(in_path) if (x.endswith(".wav") or x.endswith(".WAV"))
    ]
    target_cnt = len(tracks)
    error_list = []
    skipped_cnt = 0

    print(f"Start extracting chorus of {target_cnt} files...")
    with futures.ProcessPoolExecutor(max_workers=4) as exe:
        results = tqdm(
            exe.map(_extract_chorus, zip(tracks, repeat(in_path), repeat(out_path))),
            total=target_cnt,
        )
        for result in results:
            if result == -1:
                skipped_cnt += 1
            elif result is not True:
                error_list.append(result)

    # results = tqdm(
    #     map(_extract_chorus, zip(tracks, repeat(in_path), repeat(out_path))),
    #     total=target_cnt,
    # )
    # for result in results:
    #     if result == -1:
    #         skipped_cnt += 1
    #     elif result is not True:
    #         error_list.append(result)

    return target_cnt, error_list, skipped_cnt


def export_feature(tracks, out_path):
    error_list = []
    features = [
        "tempo",
        "zero_crossings",
        "spectral_centroid",
        "spectral_rolloff",
        "chroma",
        "mfcc",
    ]
    total_amount = len(tracks.values())
    with futures.ThreadPoolExecutor(max_workers=4) as exe:
        fs = tqdm(
            exe.map(export_track, zip(tracks.values(), repeat(out_path))),
            total=total_amount,
        )
        for done in futures.as_completed(fs):
            error_obj = done.result()
            if error_obj is not True:
                error_list.append(error_obj)

    return error_list


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


def main():
    init(autoreset=True)
    c, db_name = connect_db()
    init_db(c, db_name)
    flag = create_table(c, db_name)
    c.close()  # close connection to open new one in itdbloader
    if flag is False:  # if create_table failed
        return

    # import to DB
    config = itdbloader.get_config()
    db_loader = itdbloader.load_itdb(config)

    # convert to wav
    tracks = db_loader.itunes["Tracks"]
    total_cnt, failed, skip_cnt = convert_wav(tracks, WAV_PATH)
    print_results(total_cnt, failed, skip_cnt)

    total_cnt, failed, skip_cnt = extract_chorus(WAV_PATH, CHORUS_PATH)
    print_results(total_cnt, failed, skip_cnt)

    # export features


if __name__ == "__main__":
    freeze_support()
    main()
