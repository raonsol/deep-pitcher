from audio_process import AudioConverter
from concurrent import futures
from multiprocessing import freeze_support
from utils import print_results, get_paths, get_filename, is_dir
from itertools import repeat
from tqdm import tqdm

import os

TARGET_PATH = "D:\\chorus-22khz"
VOCAL_PATH = "D:\\vocals"
PITCH_PATH = "D:\\pitch"
N_PROCESS = 3
OVERWRITE = False


def _extract_vocal(data):
    track, path = data
    export_path = os.path.join(path, get_filename(track) + "_vocals.wav")

    # if the converted file exists, and the overwrite option is False
    if (not OVERWRITE) and os.path.exists(export_path):
        return -1

    try:
        c = AudioConverter(track)
        c.extract_vocal_by_file(path)  # input only path, not filename
        return True

    except Exception as e:
        print("Error processing file", track)
        print(e)
        return False


# TODO: NOT use map, use singleton spleeter object
def extract_vocal(in_path, out_path):
    track_paths = get_paths(in_path)
    target_cnt = len(track_paths)
    error_list = []
    skipped_cnt = 0

    print(f"Starting extract vocals for {target_cnt} files...")
    # DO NOT use multiprocessing here, because spleeter does
    results = tqdm(
        map(_extract_vocal, zip(track_paths, repeat(out_path))),
        total=target_cnt,
    )

    for result in results:
        if result == -1:
            skipped_cnt += 1
        elif result is False:
            error_list.append(result)

    return target_cnt, error_list, skipped_cnt


def _detect_pitch(data):
    track, path = data

    # # if the converted file exists, and the overwrite option is False
    if (not OVERWRITE) and os.path.exists(
        os.path.join(path, get_filename(track) + ".f0.csv")
    ):
        return -1

    try:
        c = AudioConverter(track)
        c.extract_pitch(out_path=path, method="crepe", model_size="full")
        return True

    except Exception as e:
        print("Error processing file", track)
        print(e)
        return False


def detect_pitch(in_path, out_path):
    track_paths = get_paths(in_path)
    target_cnt = len(track_paths)
    error_list = []
    skipped_cnt = 0

    print(f"Starting detecting pitch for {target_cnt} files...")
    with futures.ProcessPoolExecutor(max_workers=N_PROCESS) as exe:
        results = tqdm(
            exe.map(_detect_pitch, zip(track_paths, repeat(out_path))),
            total=target_cnt,
        )

        for result in results:
            if result == -1:
                skipped_cnt += 1
            elif result is False:
                error_list.append(result)

    return target_cnt, error_list, skipped_cnt


def main():
    in_path = TARGET_PATH
    vocal_path = VOCAL_PATH
    pitch_path = PITCH_PATH
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    is_dir(vocal_path, default="vocal")
    if not os.path.isdir(in_path):
        print("No such input directory: %s" % in_path)

    # total_cnt, errors, skipped_cnt=extract_vocal(in_path, vocal_path)
    # print_results(total_cnt, errors, skipped_cnt)

    is_dir(vocal_path, default="pitch")
    total_cnt, errors, skipped_cnt = detect_pitch(vocal_path, pitch_path)
    print_results(total_cnt, errors, skipped_cnt)


if __name__ == "__main__":
    freeze_support()
    main()
