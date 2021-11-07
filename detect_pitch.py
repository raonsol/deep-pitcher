from audio_process import AudioConverter
from multiprocessing import freeze_support
from utils import print_results, get_paths, get_filename, is_dir
from itertools import repeat
from tqdm import tqdm

import os

TARGET_PATH = "/Volumes/vault0/dataset3/test"
DEST_PATH = "/Volumes/vault0/dataset3/vocal_16"
N_PROCESS = 10
OVERWRITE_VOCAL = False

def _extract_vocal(data):
    track, path = data
    export_path = os.path.join(path, get_filename(track) + "_vocals.wav")

    # if the converted file exists, and the overwrite option is False
    if (not OVERWRITE_VOCAL) and os.path.exists(export_path):
        return -1

    try:
        c = AudioConverter(track)
        c.extract_vocal_by_file(path)   #input only path, not filename
        return True

    except Exception as e:
        print("Error processing file", track)
        print(e)
        return False

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

def main():
    in_path=TARGET_PATH
    out_path = DEST_PATH

    # TODO: path handling method in utils
    # TODO: make dest path if not exist

    if not is_dir(out_path):
        print("Using default path...")
        out_path = os.path.join(os.getcwd(), "vocal")
        os.mkdir(out_path)

    if not os.path.isdir(TARGET_PATH):
        print("No such input directory: %s" % TARGET_PATH)

    total_cnt, errors, skipped_cnt=extract_vocal(in_path, out_path)
    print_results(total_cnt, errors, skipped_cnt)

if __name__ == "__main__":
    freeze_support()
    main()