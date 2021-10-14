from pydub import AudioSegment
import tempfile
from pydub.utils import mediainfo
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
import errno
import librosa
import crepe
import pychorus

# 경로와 포맷 지정
ROOT_PATH = os.path.join(os.path.expanduser("~"), "audio_root/")
BASE_PATH = ROOT_PATH + "original"
OUT_PATH = ROOT_PATH + "out"
OUT_VOCAL_PATH = ROOT_PATH + "out_v"
FORMAT = "wav"
SAMPLE_RATE = 16000


class AudioConverter:
    T_SEC = 1000
    T_MIN = T_SEC * 60

    @profile
    def __init__(self, src_path, process_path="/tmp"):
        """
        :param src_path: audio file path
        :type src_path: string
        :param process_path: path for store wav files while processing, default is "/tmp"
        :type process_path: string
        """
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

    def get_features(self, feature_list):
        """analyzes audio file and returns feature values

        :param feature_list: list of features to export
        :type feature_list: list
        :returns: dictionary of features
        :rtype: dict
        """

        y_harm, y_perc = librosa.effects.hpss(self.y)
        result = {}

        if ("tempo" or "BPM") in feature_list:
            tempo, _ = librosa.beat.beat_track(self.y, sr=self.sr)
            result["bpm"] = tempo
        if "zero_crossings" in feature_list:
            zc = librosa.zero_crossings(self.y, pad=False)
            result["zero_crossings"] = zc
        if "spectral_centroid" in feature_list:
            sp_c = librosa.feature.spectral_centroid(self.y, sr=self.sr)[0]
            result["spectral_centroid"] = sp_c
        if "spectral_rolloff" in feature_list:
            sp_r = librosa.feature.spectral_rolloff(self.y, sr=self.sr)[0]
            result["spectral_rolloff"] = sp_r
        if "chroma" in feature_list:
            chromagram = librosa.feature.chroma_stft(self.y, sr=self.sr, hop_length=512)
            result["chroma"] = chromagram
        if "mfcc" in feature_list:
            mfccs = librosa.feature.mfcc(self.y, sr=self.sr, n_mfcc=20)
            result["mfcc"] = mfccs

        return result

    def export_features(self, features):
        return

    def extract_vocal_by_file(self, out_path, out_name, option=2):
        """
        Separates the vocal and accompaniments and export it into audio file

        :param out_path: Output path
        :param out_name: Output name
        :param option: (optional) number of channels that wants to separate, default=2
        """
        # 2stems: vocal + background music
        separator = Separator("spleeter:%sstems" % str(option))
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
        separator = Separator("spleeter:%sstems" % str(option))
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

    def pitch_detect_alt(self):
        model = hub.load("./spice_2")
        audio_sample = self.src.get_array_of_samples()
        model_output = model.signatures["serving_default"](
            tf.constant(audio_sample, tf.float32)
        )
        pitch_outputs = model_output["pitch"]
        uncertainty_outputs = model_output["uncertainty"]
        confidence_outputs = 1.0 - uncertainty_outputs

    @profile
    def detect_chorus(self):
        """
        :return: Time in seconds of the start of the chorus
        :rtype: float
        """
        # pychorus.create_chroma(self.y)
        s = np.abs(librosa.stft(self.y, n_fft=2 ** 14)) ** 2
        chroma = librosa.feature.chroma_stft(S=s, sr=self.sr)
        chorus_start_sec = pychorus.find_chorus(chroma, self.sr, self.duration, 5)

        # if the estimated chrous sec seems incorrect, we use default value
        if not chorus_start_sec or self.duration - chorus_start_sec < 30:
            chorus_start_sec = 60

        return chorus_start_sec

    def get_name(self, n, extend=""):
        """
        create file name to export

        :param n: 순번 값
        :param extend: (optional) 덧붙일 문자열
        :returns: file name without format
        :rtype: String
        """
        n_artist = self.meta["artist"]
        n_title = self.meta["title"]

        # remove charactors that might be problem
        remove_list = ",?!()&`_'" + '"'
        for x in range(len(remove_list)):
            n_artist = n_artist.replace(remove_list[x], "")
            n_title = n_title.replace(remove_list[x], "")

        return n + "_" + n_artist + "_" + n_title + extend

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
                    # 이름 출력에 문제있는 경우
                    if e.errno and e.errno == errno.EISDIR:
                        print("Directory Error:")
                        print(e)
                    # 그 밖의 경우
                    else:
                        print("Error processing ", n, "\n", e)

    pitch_result.to_csv("pitch_result.csv", mode="w")
    print("Pitch result saved")
