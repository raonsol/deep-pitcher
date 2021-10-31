# %% [markdown]
# # EDA for visualization
# %%
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
import numpy
from sklearn.preprocessing import minmax_scale

# %% [markdown]
# ## Visualize one target
#
# TARGET: 10888 (에픽하이 - 우산 (feat.윤하))

# %%
SOURCE_PATH = "/Volumes/vault0/dataset3/chorus-22khz"
FEATURE_CSV_PATH = "./result_22000hz.csv"
TARGET = "10888.wav"

target = os.path.join(SOURCE_PATH, TARGET)
x, sr = librosa.load(target)

print(f"Length: {x.shape}, Sample Rate: {sr}")
ipd.Audio(target)

# %% [markdown]
# ### WavePlot

# %%
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr, alpha=0.8)

# %% [markdown]
# ### Spectrogram

# %%
x_stft = librosa.stft(x)
x_db = librosa.amplitude_to_db(abs(x_stft))
plt.figure(figsize=(14, 5))
librosa.display.specshow(x_db, sr=sr, x_axis="time", y_axis="hz")
plt.colorbar(format="%+2.0f dB")

# %% [markdown]
# ### RMSE (Root-square Minimum Energy)
# $$ \sqrt{ \frac{1}{N} \sum_n \left| x(n) \right|^2 } $$

# %%
n_fft = 2048
frame_length = 512
hop_length = 256


# %%
energy = numpy.array(
    [sum(abs(x[i : i + frame_length] ** 2)) for i in range(0, len(x), 256)]
)
rmse = librosa.feature.rms(
    x, frame_length=frame_length, hop_length=hop_length, center=True
)[0]

t = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=hop_length)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, minmax_scale(energy, axis=0), label="Energy", linestyle="dashed")
plt.plot(t, rmse, label="RMSE")
plt.legend()

# %% [markdown]
# ## Zero Crossing Rate
#
# Indicates the number of times that a signal crosses the horizontal axis.

# %%
zero_crossings = librosa.zero_crossings(x, pad=False)
zcr = librosa.feature.zero_crossing_rate(x)
print("Zero Crossing count: " + str(sum(zero_crossings)))
plt.figure(figsize=(14, 5))
plt.plot(zcr[0], label="ZCR")
plt.legend()

# %% [markdown]
# ## Chroma

# %%
chroma_stft = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
chroma_cqt = librosa.feature.chroma_cqt(x, sr=sr, hop_length=hop_length)
chroma_cens = librosa.feature.chroma_cens(x, sr=sr, hop_length=hop_length)

plt.figure(figsize=(15, 4))
plt.title("chroma_stft", fontsize=16)
librosa.display.specshow(
    chroma_stft, x_axis="time", y_axis="chroma", hop_length=hop_length, cmap="coolwarm"
)
plt.figure(figsize=(15, 4))
plt.title("chroma_cqt", fontsize=16)
librosa.display.specshow(
    chroma_cqt, x_axis="time", y_axis="chroma", hop_length=hop_length, cmap="coolwarm"
)
plt.title("chroma_cens", fontsize=16)
plt.figure(figsize=(15, 4))
librosa.display.specshow(
    chroma_cens, x_axis="time", y_axis="chroma", hop_length=hop_length, cmap="coolwarm"
)

# %% [markdown]
# ## Spectral Features

# %%
# Spectral Centroids
spectral_centroids = librosa.feature.spectral_centroid(x + 0.01, sr=sr)[0]
spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr=sr)[0]
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

plt.figure(figsize=(20, 4))
plt.title("Spectral centroids and rolloff", fontsize=16)
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, minmax_scale(spectral_rolloff), color="r")
plt.plot(t, minmax_scale(spectral_centroids), color="g")
plt.legend(("spectral rolloff", "spectral centroids"))

# Spectral Bandwidth
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr, p=4)[0]

plt.figure(figsize=(20, 4))
plt.title("Spectral Bandwidth", fontsize=16)
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, minmax_scale(spectral_bandwidth_2), color="r")
plt.plot(t, minmax_scale(spectral_bandwidth_3), color="g")
plt.plot(t, minmax_scale(spectral_bandwidth_4), color="y")
plt.legend(("p = 2", "p = 3", "p = 4"))

# %% [markdown]
# ## MFCC (Mel Frequency Cepstral Coefficient)

# %%
MFCCs = librosa.feature.mfcc(x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=20)
plt.figure(figsize=(14, 7))
librosa.display.specshow(
    MFCCs, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz"
)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.show()

# %% [markdown]
# ## Tempo

# %%
tempo = librosa.beat.tempo(x, sr=sr)
T = len(x) / float(sr)
seconds_per_beat = 60.0 / tempo[0]
beat_times = numpy.arange(0, T, seconds_per_beat)

plt.figure(figsize=(14, 5))
plt.title("Tempo")
librosa.display.waveplot(x, alpha=0.8)
plt.vlines(beat_times, -1, 1, color="r")

clicks = librosa.clicks(beat_times, sr, length=len(x))
ipd.Audio(x + clicks, rate=sr)

# %% [markdown]
# ## EDA for dataset

# %%
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data(path):
    data = pd.read_csv(path, index_col=[0], header=[0, 1, 2])
    return data


# 데이터 불러오기
data = load_data("result_22000hz.csv")
pd.set_option("display.max_columns", None)
data.head()
data_labels = ["_".join(a) for a in data.columns.to_flat_index()]
features = data.columns.levels[0].tolist()
moments = data.columns.levels[1].tolist()
print(f"Features: {features}")
print(f"Moments: {moments}")
data.head()


# %%
# set style
plt.rcParams["figure.figsize"] = (15, 30)
plt.rcParams["font.size"] = 12


# %%
def make_boxplots(feature, moments):
    n = len(moments)
    fig, ax = plt.subplots(
        int(n / 2) + n % 2, 2, figsize=(17, 15), constrained_layout=True
    )
    plt.suptitle(feature, fontsize=20)
    for i in range(n):
        ax[int(i / 2)][i % 2].boxplot(data[feature, moments[i]], patch_artist=True)
        ax[int(i / 2)][i % 2].set_title(moments[i])


# %%
make_boxplots("chroma_cens", moments)
make_boxplots("mfcc", moments)
make_boxplots("tonnetz", moments)


# %%
fig, ax = plt.subplots(1, 1, figsize=(17, 7))
ax.hist(data["tempo", "mean"], bins=50)
ax.legend("tempo")
ax.set_title("tempo, mean")
plt.show()


# %%
fig, ax = plt.subplots(1, 1, figsize=(17, 7))
ax.hist(data["tonnetz", "mean"], bins=50)
# ax.legend()
ax.set_title("tonnetz, mean")
plt.show()
