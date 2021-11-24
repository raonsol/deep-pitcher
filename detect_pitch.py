from audio_process import AudioConverter
from concurrent import futures
from multiprocessing import freeze_support
from utils import print_results, get_paths, get_filename, is_dir
import features
from itertools import repeat
from tqdm import tqdm
import pandas as pd
import numpy as np

import os

TARGET_PATH = "/Volumes/vault0/dataset3/test"
VOCAL_PATH = "/Volumes/vault0/dataset3/vocals"
PITCH_PATH = "/Volumes/vault0/dataset3/pitch"
N_PROCESS = 3
OVERWRITE_VOCAL = False

def _extract_vocal(data):
    track, path = data
    export_path = os.path.join(path, get_filename(track) + "_vocals.wav")

    # if the converted file exists, and the overwrite option is False
    if (not OVERWRITE_VOCAL) and os.path.exists(export_path):
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


def _refine_pitch(track):
    """refine pitch to handle outliers
    
    :param path: csv path to read
    :return pitch_mean: mean value of the pitch
    :return pitch_max: max value of the pitch
    :return pitch_min: min value of the pitch
    """
    predict_val=pd.read_csv(track)

    # drop values except C2(65Hz) ~ G5(784Hz)
    predict_val.loc[predict_val.frequency<65, 'frequency']=None
    predict_val.loc[predict_val.frequency>784, 'frequency']=None

    # drop values if confidance value is under 0.5
    predict_val.loc[predict_val.confidence<0.5, 'frequency']=None


    pitch_mean=predict_val['frequency'].mean()
    pitch_max=predict_val['frequency'].max()
    pitch_min=predict_val['frequency'].min()

    return pitch_mean, pitch_max, pitch_min

def refine_pitch(in_path):
    """refine pitch in the directory
    
    :param in_path: csv path to read
    :return data_df: mean, max, min of pitch data
    :rtype data_df: pandas.DataFrame
    
    """
    track_paths = get_paths(in_path)
    target_cnt = len(track_paths)
    error_list = []
    result_list=[]
    idx=[]

    print(f"Starting refining pitch for {target_cnt} files...")
    for target in tqdm(track_paths):
        try:
            idx.append(get_filename(target).replace(".f0", ""))
            result=_refine_pitch(target)
            result_list.append(result)
        except Exception:
            error_list.append(target)

    result_df = pd.DataFrame(result_list, columns=["mean", "max", "min"], index=idx)
    
    return result_df, target_cnt, error_list
    

def _detect_pitch(data):
    track, path = data

    # # if the converted file exists, and the overwrite option is False
    if (not OVERWRITE_VOCAL) and os.path.exists(
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


def extract_pitch_features(in_path):
    tracks = get_paths(in_path)
    target_cnt = len(tracks)
    error_list = []
    # skipped_cnt = 0

    feature_list = dict(
        tonnetz=6,
        # tempo=1,
        # rms=1,
        # zcr=1,
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
    idxs = [get_filename(track) for track in tracks]
    cols = features.get_columns(feature_list, moments)
    result_df = pd.DataFrame(index=idxs, columns=cols, dtype=np.float32)

    print(f"Starting extracting features for {target_cnt} files...")
    with futures.ProcessPoolExecutor() as exe:
        results = tqdm(
            exe.map(features.extract_features, zip(tracks, repeat(feature_list), repeat(moments))),
            total=target_cnt,
        )

        for result in results:
            error_obj, y = result
            if error_obj is not True:
                error_list.append(y.name)
            else:
                result_df.loc[y.name] = y
    
    # #for debugging
    # results = tqdm(
    #         map(features.extract_features, zip(tracks, repeat(feature_list), repeat(moments))),
    #         total=target_cnt,
    #     )  

    # for result in results:
    #     error_obj, y = result
    #     if error_obj is not True:
    #         error_list.append(y.name)
    #     else:
    #         result_df.loc[y.name] = y
    

    return result_df, target_cnt, error_list


def main():
    in_path = TARGET_PATH
    vocal_path = VOCAL_PATH
    pitch_path = PITCH_PATH

    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    is_dir(vocal_path, default="vocal")
    if not os.path.isdir(in_path):
        print("No such input directory: %s" % in_path)

    total_cnt, errors, skipped_cnt=extract_vocal(in_path, vocal_path)
    print_results(total_cnt, errors, skipped_cnt)

    is_dir(vocal_path, default="pitch")
    total_cnt, errors, skipped_cnt = detect_pitch(vocal_path, pitch_path)
    print_results(total_cnt, errors, skipped_cnt)
    
    pitch_path=PITCH_PATH
    if not os.path.isdir(pitch_path):
        print("No such input directory: %s" % pitch_path)

    pitch_df, error_list, skipped_cnt=refine_pitch(pitch_path)
    pitch_df.to_csv("./results/pitch_result.csv")

    vocal_features, n, error_list = extract_pitch_features(vocal_path)
    vocal_features.to_csv("./results/vocal_features.csv")
    print_results(n, error_list, 0)
    

if __name__ == "__main__":
    freeze_support()
    main()
