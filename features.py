import os
from concurrent import futures
from itertools import repeat
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import colorama
from colorama import Fore, Style
from tqdm import tqdm
from line_profiler_pycharm import profile

from audio_process import AudioConverter
from utils import print_results, get_paths, get_filename, is_dir

TARGET_PATH = "/Volumes/vault0/dataset3/chorus-22khz"
DEST_PATH = "/Volumes/vault0/dataset3/feature"
N_PROCESS = 10


def get_columns(feature_list, moments, prefix=None):
    columns = []

    # ex: (mfccs, mean, 01), (mfccs, mean, 02), (mfccs, max, 01), ...
    for name, size in feature_list.items():
        if prefix:
            name=prefix+"_"+name

        if name == "tempo":
            it = ((name, "mean", "01"),)
            columns.extend(it)
        else:
            for moment in moments:
                it = ((name, moment, "{:02d}".format(i + 1)) for i in range(size))
                columns.extend(it)

    names = ("feature", "statistics", "number")
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


# @profile
def extract_features(data):
    track, features, moments = data
    feature = {}
    idx = get_columns(features, moments)
    try:
        c = AudioConverter(track)
        feature = c.extract_features(features, moments, idx)
        success = True
    except Exception:
        success = track

    return success, feature


#TODO: skip if already exists
def extract_features_parallel(in_path, feature_list, moments):
    tracks = get_paths(in_path)
    n_total = len(tracks)
    idxs = [get_filename(track) for track in tracks]
    cols = get_columns(feature_list, moments)
    feature_result = pd.DataFrame(index=idxs, columns=cols, dtype=np.float32)
    error_list = []

    # use multiprocessing
    print(f"Starting extract features for {n_total} files...")
    with futures.ProcessPoolExecutor(max_workers=N_PROCESS) as exe:
        results = tqdm(
            exe.map(
                extract_features,
                zip(tracks, repeat(feature_list), repeat(moments)),
            ),
            total=n_total,
        )
        for result in results:
            error_obj, y = result
            if error_obj is not True:
                error_list.append(error_obj)
            else:
                feature_result.loc[y.name] = y

    # single process run for test
    # results = tqdm(
    #     map(
    #         extract_features,
    #         zip(tracks, repeat(feature_list), repeat(moments)),
    #     ),
    #     total=n_total,
    # )
    # for result in results:
    #     error_obj, y = result
    #     if error_obj is not True:
    #         error_list.append(error_obj)
    #     else:
    #         feature_result.loc[y.name] = y

    return feature_result, n_total, error_list


# @profile
def main():
    colorama.init(autoreset=True)

    # 75 features
    feature_list = dict(
        tonnetz=6,
        tempo=1,
        rms=1,
        zcr=1,
        spectral_centroid=1,
        spectral_bandwidth=1,
        spectral_contrast=7,
        spectral_rolloff=1,
        chroma_stft=12,
        chroma_cqt=12,
        chroma_cens=12,
        mfcc=20,
    )
    moments = ["mean", "std", "skew", "kurtosis", "median", "min", "max"]

    target_path = TARGET_PATH
    out_path = DEST_PATH
    if not is_dir(out_path):
        print("Using default path...")
        out_path = os.path.join(os.getcwd(), "feature")
        os.mkdir(out_path)

    feature_result, n_total, error_list = extract_features_parallel(
        target_path, feature_list, moments
    )
    print_results(n_total, error_list, 0, print_failed=True)

    # save to csv
    print("Destination path:" + out_path)
    print("Saving result to csv... ", end="")
    try:
        feature_result.to_csv(
            os.path.join(out_path, "result.csv"), float_format="%.{}e".format(5)
        )
    except Exception as e:
        print()
        print(Fore.RED + "Error: " + repr(e))
    print(Fore.GREEN + "Success")


if __name__ == "__main__":
    freeze_support()
    main()
