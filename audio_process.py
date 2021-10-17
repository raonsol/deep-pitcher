import contextlib

from pydub import AudioSegment, effects
import tempfile
from pydub.utils import mediainfo
from scipy import stats
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
from multiprocessing import freeze_support
from line_profiler_pycharm import profile
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os
import librosa
import crepe
import pychorus

# 경로와 포맷 지정
from utils import get_filename

ROOT_PATH = os.path.join(os.path.expanduser("~"), "audio_root/")
BASE_PATH = ROOT_PATH + "original"
OUT_PATH = ROOT_PATH + "out"
OUT_VOCAL_PATH = ROOT_PATH + "out_v"
FORMAT = "wav"
SAMPLE_RATE = 16000


class AudioConverter:
    T_SEC = 1000
    T_MIN = T_SEC * 60

    # @profile
    def __init__(self, src_path, process_path="/tmp"):
        """
        :param src_path: audio file path
        :type src_path: string
        :param process_path: path for store wav files while processing, default is "/tmp"
        :type process_path: string
        """
        # TODO: src에 경로나 음악파일 중 아무거나 집어넣어도 되도록 구현
        self.src_path = src_path
        self.process_path = process_path
        self.src = AudioSegment.from_file(src_path)
        self.y, self.sr = librosa.load(
            src_path, sr=16000, mono=True
        )  # resample sr to 16kHz, use None to load with original sr
        self.duration = self.y.shape[0] / float(self.sr)
        self.meta = mediainfo(src_path).get("TAG", None)

    def cut_audio(self, sec_start, sec_dur):
        """
        cut loaded audio from sec_start with sec_dur

        :param sec_start: 자르기 시작할 시간 (단위: 초)
        :param sec_dur: 자를 길이 (단위: 초)
        """
        # TODO: fill left time with blank data to match the sec_dur (is it necessary?)
        # if sec_start + sec_dur > self.duration:
        #     pass
        self.src = self.src[
            self.T_SEC * sec_start : self.T_SEC * sec_start + self.T_SEC * sec_dur
        ]

    # @profile
    def get_features(self, features, moments, idx):
        """analyzes audio file and returns feature values

        :param moments: moments to extract (ex: mean, max, median...)
        :type moments: tuple
        :param features: list and amount of features to export (ex: mfccs=20, bpm=1...)
        :type features: dict
        :param idx: column to use for return Series
        :type idx: pd.MultiIndex
        :returns: Series of features
        :rtype: pd.Series
        """

        def feature_stats(name, values):
            if name == "tempo":
                result[name, "mean"] = values
                return

            if "mean" in moments:
                result[name, "mean"] = np.mean(values, axis=1)
            if "std" in moments:
                result[name, "std"] = np.std(values, axis=1)
            if "skew" in moments:
                result[name, "skew"] = stats.skew(values, axis=1)
            if "kurtosis" in moments:
                result[name, "kurtosis"] = stats.kurtosis(values, axis=1)
            if "median" in moments:
                result[name, "median"] = np.median(values, axis=1)
            if "min" in moments:
                result[name, "min"] = np.min(values, axis=1)
            if "max" in moments:
                result[name, "max"] = np.max(values, axis=1)

        result = pd.Series(
            index=idx, dtype=np.float32, name=get_filename(self.src_path)
        )
        y_harm, y_perc = librosa.effects.hpss(self.y)

        # cqt
        cqt = np.abs(
            librosa.cqt(
                self.y,
                sr=self.sr,
                hop_length=512,
                bins_per_octave=12,
                n_bins=7 * 12,
                tuning=None,
            )
        )
        assert cqt.shape[0] == 7 * 12
        assert (
            np.ceil(len(self.y) / 512) <= cqt.shape[1] <= np.ceil(len(self.y) / 512) + 1
        )

        # stft
        stft = np.abs(librosa.stft(self.y, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert (
            np.ceil(len(self.y) / 512)
            <= stft.shape[1]
            <= np.ceil(len(self.y) / 512) + 1
        )

        if "tempo" in features:
            # use beat.plp to get stats
            tempo = librosa.beat.tempo(y_perc, sr=self.sr)
            feature_stats("tempo", tempo)
        if "tonnetz" in features:
            f = librosa.feature.tonnetz(
                chroma=librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
            )
            feature_stats("tonnetz", f)
        if ("rms" or "rmse") in features:
            f = librosa.feature.rms(S=stft)
            feature_stats("rms", f)
        if "zcr" in features:
            x = librosa.feature.zero_crossing_rate(self.y, pad=False)
            feature_stats("zcr", x)

        if "spectral_centroid" in features:
            x = librosa.feature.spectral_centroid(S=stft)
            feature_stats("spectral_centroid", x)
        if "spectral_bandwidth" in features:
            x = librosa.feature.spectral_bandwidth(S=stft)
            feature_stats("spectral_bandwidth", x)
        if "spectral_contrast" in features:
            x = librosa.feature.spectral_contrast(S=stft)
            feature_stats("spectral_contrast", x)
        if "spectral_rolloff" in features:
            x = librosa.feature.spectral_rolloff(S=stft)
            feature_stats("spectral_rolloff", x)

        if "chroma_stft" in features:
            x = librosa.feature.chroma_stft(
                S=stft ** 2, n_chroma=features["chroma_stft"]
            )
            feature_stats("chroma_stft", x)
        if "chroma_cqt" in features:
            x = librosa.feature.chroma_cqt(
                C=cqt, n_chroma=features["chroma_cqt"], n_octaves=7
            )
            feature_stats("chroma_cqt", x)
        if "chroma_cens" in features:
            x = librosa.feature.chroma_cens(
                C=cqt, n_chroma=features["chroma_cens"], n_octaves=7
            )
            feature_stats("chroma_cens", x)

        if "mfcc" in features:
            mel = librosa.feature.melspectrogram(sr=self.sr, S=stft ** 2)
            # apply log scaling (dB) for mfcc
            x = librosa.feature.mfcc(
                S=librosa.power_to_db(mel), n_mfcc=features["mfcc"]
            )
            feature_stats("mfcc", x)

        return result

    def extract_vocal_by_file(self, out_path, out_name, option=2):
        """
        Separates the vocal and accompaniments and export it into audio file

        :param out_path: Output path
        :param out_name: Output name
        :param option: (optional) number of channels that wants to separate, default=2
        """
        # 2stems: vocal + background music
        separator = Separator("spleeter:%sstems-16kHz" % str(option))
        separator.separate_to_file(
            self.src_path, out_path, filename_format=out_name + "_{instrument}.{codec}"
        )

    def extract_vocal(self, option=2):
        """
        Separates the vocal and accompaniments and returns the waveform

        :param option: 분리할 채널 수, default=2
        :returns: Dictionary that contains numpy array ({'vocals'}, {'accompaniment'})
        :rtype: dict
        """
        freeze_support()
        # 2stems: vocal + background music
        separator = Separator("spleeter:%sstems-16kHz" % str(option))
        waveform, _ = AudioAdapter.default().load(
            self.src_path, sample_rate=SAMPLE_RATE
        )
        prediction = separator.separate(waveform)

        return prediction

    def detect_pitch(self, out_name="out", model_size="small"):
        """detect pitch values

        :param out_name: (optional) 내부에서 임시로 사용할 파일명
        :param model_size: tiny', 'small', 'medium', 'large', 'full'

        :returns Tuple: (time: np.ndarray [shape=(T,)]
        :returns frequency: np.ndarray [shape=(T,)]
        :returns activation: np.ndarray [shape=(T, 360)]
        """

        with tempfile.TemporaryDirectory() as temp_path:
            # temp_path.{out_name}_vocals.wav 이름으로 임시공간에 출력
            self.extract_vocal_by_file(temp_path, out_name)
            temp_path_out = os.path.join(temp_path, out_name + "_vocals.wav")

            # 임시공간에 있는 파일로 pitch detect
            prediction = crepe.process_file(
                temp_path_out,
                output=temp_path,
                model_capacity=model_size,
                save_activation=False,
                save_plot=False,
                plot_voicing=False,
                step_size=100,
                viterbi=False,
            )
            predicted_path = os.path.join(temp_path, out_name + "_vocals.f0.csv")
            predict_val = np.transpose(
                np.genfromtxt(predicted_path, delimiter=",", dtype=float, skip_header=1)
            )

        return predict_val[0], predict_val[1], predict_val[2]

    def detect_pitch_alt(self):
        model = hub.load("./spice_2")
        audio_sample = self.src.get_array_of_samples()
        model_output = model.signatures["serving_default"](
            tf.constant(audio_sample, tf.float32)
        )
        pitch_outputs = model_output["pitch"]
        uncertainty_outputs = model_output["uncertainty"]
        confidence_outputs = 1.0 - uncertainty_outputs

    # @profile
    def detect_chorus(self):
        """
        :return: Time in seconds of the start of the chorus
        :rtype: float
        """
        # pychorus.create_chroma(self.y)
        s = np.abs(librosa.stft(self.y, n_fft=2 ** 14)) ** 2
        chroma = librosa.feature.chroma_stft(S=s, sr=self.sr)

        # mute error messages of find_chorus()
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            chorus_start_sec = pychorus.find_chorus(chroma, self.sr, self.duration, 10)
            flag = True

        # if the estimated chrous sec seems incorrect, we use default value
        if not chorus_start_sec or self.duration - chorus_start_sec < 30:
            chorus_start_sec = 60
            flag = False

        return chorus_start_sec, flag

    def get_name(self, option=0, extend=""):
        """
        create file name to export

        :param option: 0 for use original name, 1 for {artist_title} format
        :param extend: (optional) 덧붙일 문자열
        :returns: file name without format
        :rtype: String
        """
        if option:
            n_artist = self.meta["artist"]
            n_title = self.meta["title"]

            # remove charactors that might be problem
            remove_list = ",?!()&`_'" + '"'
            for x in range(len(remove_list)):
                n_artist = n_artist.replace(remove_list[x], "")
                n_title = n_title.replace(remove_list[x], "")

            return n + "_" + n_artist + "_" + n_title + extend

        else:
            return os.path.splitext(os.path.basename(self.src_path))[0]

    def normalize(self):
        self.src = effects.normalize(self.src)

    def export(self, export_path=None, format="wav", sample_rate=SAMPLE_RATE):
        """export loaded audio to selected format and path

        :param export_path: path to export file, default is process_path of AudioConverter class
        :param format: format of audio, default is wav
        :param sample_rate: target sample rate to export
        """

        if export_path is None:
            export_path = self.process_path
        self.src.export(
            export_path,
            format=format,
            tags=self.meta,
            parameters=["-ar", str(sample_rate), "-ac", "1"],
        )


if __name__ == "__main__":
    # Dataframe for saving result of pitch detection
    pitch_column = ["filename", "pitch_mean", "pitch_max"]
    pitch_result = pd.DataFrame([], columns=pitch_column)

    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)

    if not os.path.isdir(BASE_PATH):
        print("No such input directory: %s" % BASE_PATH)

    # BASE_PATH 내의 오디오 파일들에 대해 변환 수행
    else:
        for root, dirs, files in os.walk(BASE_PATH):

            if ".DS_Store" in files:
                files.remove(".DS_Store")

            # 파일 처리
            for n, file in enumerate(tqdm(files)):
                n = str(n).zfill(4)
                labeling, ext = os.path.splitext(file)
                target_path = os.path.join(root, file)
                meta = mediainfo(target_path).get("TAG", None)

                # import audio file
                ac = AudioConverter(target_path)

                # set names
                out_name = ac.get_name(n)
                out_name_f = out_name + "." + FORMAT
                out_path = os.path.join(OUT_PATH, out_name_f)
                out_path_v = os.path.join(OUT_VOCAL_PATH, out_name_f)

                """
        # 이름 출력에 문제없고 이미 변환된 파일이 out폴더에 존재할 경우 변환 SKIP
        if out_name and os.path.isfile(out_path):
          print("Skipping", out_name)
          continue
        """
                # seperate vocal and detect pitch
                pitch_time, pitch_val, pitch_act = ac.detect_pitch(out_name)

                pitch_mean = pitch_val.mean()
                pitch_max = pitch_val.max()

                r = {
                    "filename": out_name_f,
                    "pitch_mean": pitch_mean,
                    "pitch_max": pitch_max,
                }
                pitch_result = pitch_result.append(r, ignore_index=True)

                # convert & export wav
                ac.cut_audio(60, 60)
                try:
                    ac.export(out_path, format=FORMAT)
                except TypeError as t:
                    print("TypeError in parameter:", t)
                except Exception as e:
                    print("Error processing ", n, "\n", e)

    pitch_result.to_csv("pitch_result.csv", mode="w")
    print("Pitch result saved")
