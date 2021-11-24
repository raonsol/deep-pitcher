import os
from multiprocessing import freeze_support
from concurrent import futures
from itertools import repeat
from urllib.parse import unquote

import pandas as pd
from colorama import Fore
from line_profiler_pycharm import profile
from tqdm import tqdm

from audio_process import AudioConverter
from utils import yes_or_no, print_results, get_paths, is_dir, get_filename

WAV_PATH = "/Volumes/vault0/dataset3/wav-22khz"
CHORUS_PATH = "/Volumes/vault0/dataset3/chorus-22khz"
WAV_OVERWRITE = False
CHORUS_OVERWRITE = False
N_PROCESS = 10
SAMPLE_RATE = 22050


def export_track(data):
    """Export the audio track into WAV file

    :param data: tuple that contains (dict of track info, export path)
    :type data: tuple
    :return: True if export was successful
    :rtype: bool
    """
    track, path = data
    track_path = unquote(track["Location"].replace("file://", ""))
    file_name = str(track["Track_ID"]) + ".wav"
    export_path = os.path.join(path, file_name)

    # if the converted file exists, and the overwrite option is False
    if (not WAV_OVERWRITE) and os.path.exists(export_path):
        return -1

    try:
        c = AudioConverter(track_path, export_path)
        c.export(format="wav", sample_rate=SAMPLE_RATE)
        return 1
    except:
        return track_path


def convert_wav(tracks, out_path):
    # TODO: create option to save full wav file or not

    if not is_dir(out_path):
        print("Using default path...")
        out_path = os.path.join(os.getcwd(), "prep")
        os.mkdir(out_path)

    error_list = []
    skip_cnt = 0
    target_cnt = len(tracks)
    print()
    print(f"Start converting {target_cnt} files...")
    with futures.ProcessPoolExecutor(max_workers=N_PROCESS) as exe:
        results = tqdm(
            exe.map(export_track, zip(tracks, repeat(out_path))),
            total=target_cnt,
        )
        for result in results:
            if result == -1:
                skip_cnt += 1
            elif result != 1:
                error_list.append(result)

    return target_cnt, error_list, skip_cnt


# @profile
def _extract_chorus(data):
    input_path, dest_path = data
    export_path = os.path.join(dest_path, get_filename(input_path) + ".wav")

    # skip duplicate if option is on
    if not CHORUS_OVERWRITE and os.path.exists(export_path):
        return -1
    try:
        c = AudioConverter(input_path, export_path)
        chorus_sec, success = c.detect_chorus()
        c.cut_audio(chorus_sec, 30)
        c.normalize()
        c.export(format="wav", sample_rate=SAMPLE_RATE)

        if not success:
            return False
        else:
            return True
    except:
        return input_path


# @profile
def extract_chorus(in_path, out_path):
    if not os.path.isdir(out_path):
        print(Fore.RED + "chorus extract path is not exist. ", end="")
        if yes_or_no("Would you like to make one?"):
            os.mkdir(out_path)
        else:
            print("Using default path...")
            out_path = "/tmp"
    track_paths = get_paths(in_path)
    target_cnt = len(track_paths)
    error_list = []
    skipped_cnt = failed_cnt = 0

    print(f"Start extracting chorus of {target_cnt} files...")
    with futures.ProcessPoolExecutor(max_workers=N_PROCESS) as exe:
        results = tqdm(
            exe.map(_extract_chorus, zip(track_paths, repeat(out_path))),
            total=target_cnt,
        )
        for result in results:
            if result == -1:
                skipped_cnt += 1
            elif result is False:
                failed_cnt += 1
            elif result is not True:
                error_list.append(result)

    # for test in single process
    # results = tqdm(
    #     map(_extract_chorus, zip(track_paths, repeat(out_path))),
    #     total=target_cnt,
    # )
    # for result in results:
    #     if result == -1:
    #         skipped_cnt += 1
    #     elif result is False:
    #         failed_cnt += 1
    #     elif result is not True:
    #         error_list.append(result)

    return target_cnt, error_list, skipped_cnt, failed_cnt


def main():
    # read tracks and convert to wav
    tracks = pd.read_csv(os.path.join("./result", "itdb_tracks.csv")).to_dict(
        "records"
    )
    total_cnt, failed, skip_cnt = convert_wav(tracks, WAV_PATH)
    print_results(total_cnt, failed, skip_cnt)

    # preprocess
    # normalize + cut 30sec of chorus part
    total_cnt, errors, skip_cnt, failed_cnt = extract_chorus(WAV_PATH, CHORUS_PATH)
    print_results(total_cnt, errors, skip_cnt)
    if failed_cnt > 0:
        print(f"{failed_cnt} items failed to estimate, used default time (60s)")


if __name__ == "__main__":
    freeze_support()
    main()
