from concurrent import futures
from itertools import repeat

from tqdm import tqdm

from preprocessing import export_track


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
