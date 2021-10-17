import contextlib
import os
from multiprocessing import freeze_support
from concurrent import futures
from itertools import repeat
from urllib.parse import unquote

import pandas as pd
from colorama import Fore
from line_profiler_pycharm import profile
from tqdm import tqdm

from audio_util import AudioConverter
from utils import yes_or_no, print_results

WAV_PATH = "/Volumes/vault0/dataset3/wav"
CHORUS_PATH = "/Volumes/vault0/dataset3/chorus"
WAV_OVERWRITE = False
CHORUS_OVERWRITE = False


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
    target_cnt = len(tracks)
    print()
    print(f"Start converting {target_cnt} files...")
    with futures.ProcessPoolExecutor(max_workers=4) as exe:
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


def main():
    # read tracks
    tracks = pd.read_csv(os.path.join("./result_csv", "itdb_tracks.csv")).to_dict()

    # convert to wav
    total_cnt, failed, skip_cnt = convert_wav(tracks, WAV_PATH)
    print_results(total_cnt, failed, skip_cnt)

    # preprocess
    total_cnt, failed, skip_cnt = extract_chorus(WAV_PATH, CHORUS_PATH)
    print_results(total_cnt, failed, skip_cnt)


if __name__ == "__main__":
    freeze_support()
    main()