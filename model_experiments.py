# %% [markdown]
# # Model experiments
# PCA와 t-SNE를 사용한 차원 축소의 효과에 대해 각각 비교하여 보고, clustering을 수행 후 t-SNE를 사용하여 결과를 시각화한다.

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


# %%
# Set default Matplotlib style
plt.rcParams["figure.figsize"] = (18, 13)

# %% [markdown]
# ## Elbow Method
# Using Elbow method to get optimal K-Means clustering nums

# %%
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

distortions = []
K = range(20, 401, 20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data)
    distortions.append(
        sum(np.min(cdist(data, kmeanModel.cluster_centers_, "euclidean"), axis=1))
        / data.shape[0]
    )

# Plot the elbow
plt.figure(figsize=(14, 7))
plt.plot(K, distortions, "bx-")
plt.xlabel("k")
plt.ylabel("Distortion")
plt.title("The Elbow Method showing the optimal k")
plt.show()

# %% [markdown]
# ## 0: t-SNE and PCA for raw data

# %%
data_labels = ["_".join(a) for a in data.columns.to_flat_index()]

# t-sne 모델 생성 및 수행
from sklearn.manifold import TSNE

model_tsne = TSNE(n_components=2, learning_rate=300)
time_start = time.time()
tsne = pd.DataFrame(model_tsne.fit_transform(data), columns=["x1", "x2"])
tsne.set_index(data.index, inplace=True)
print(f"t-SNE Done. Elepsed Time:{time.time()-time_start}")

# PCA 수행
from sklearn.decomposition import PCA

model_pca = PCA(n_components=2)
time_start = time.time()
pca = pd.DataFrame(model_pca.fit_transform(data), columns=["x1", "x2"])
pca.set_index(data.index, inplace=True)
print(f"PCA Done. Elepsed Time:{time.time()-time_start}")


# %%
# t-SNE와 PCA 수행 결과 2차원 공간에 출력
plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
plt.title("t-SNE for raw data", fontsize=15)
r0_plot_tsne = plt.scatter(
    tsne["x1"], tsne["x2"], c=range(0, tsne.shape[0]), cmap="viridis", alpha=0.6
)

plt.subplot(1, 2, 2)
plt.title("PCA for raw data", fontsize=15)
r0_plot_pca = plt.scatter(
    pca["x1"], pca["x2"], c=range(0, pca.shape[0]), cmap="viridis", alpha=0.6
)

# %% [markdown]
# ## 1: K-Means Clustering for raw data

# %%
from sklearn.cluster import KMeans

# 200개의 k-means clustering 모델 생성
kmeans = KMeans(n_clusters=200)
kmeans_50 = KMeans(n_clusters=50)
kemeans_30 = KMeans(n_clusters=30)


# %%
# run clustering
r1 = pd.DataFrame(kmeans.fit_predict(data), columns=["cluster"])
r1.set_index(data.index, inplace=True)

# run t-SNE
# r1_data=data.copy()
# r1_data["clusters"]=r1.values
# r1_tsne=TSNE(n_components=n_components, learning_rate=300).fit_transform(r1_data)

# plt.title("t-SNE on K-Means clustering", fontsize=15)
# plt.scatter(r1_tsne[:,0], r1_tsne[:,1], c=r1.values, cmap='viridis', alpha=0.6)
# plt.colorbar()
# plt.show()

plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
plt.title("K-Means clustering on raw data (t-SNE)", fontsize=15)
r1_plot1 = plt.scatter(tsne["x1"], tsne["x2"], c=r1.values, cmap="viridis", alpha=0.6)
plt.subplot(1, 2, 2)
plt.title("K-Means clustering on raw data (PCA)", fontsize=15)
r1_plot2 = plt.scatter(pca["x1"], pca["x2"], c=r1.values, cmap="viridis", alpha=0.6)
plt.show()

# %% [markdown]
# ## 2: K-Means clustering after t-SNE

# %%
r2 = pd.DataFrame(kmeans.fit_predict(tsne), columns=["cluster"])
r2.set_index(data.index, inplace=True)

plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
plt.title("K-Means clustering after t-SNE (t-SNE)", fontsize=15)
r2_plot1 = plt.scatter(tsne["x1"], tsne["x2"], c=r2.values, cmap="viridis", alpha=0.6)
plt.subplot(1, 2, 2)
plt.title("K-Means clustering after t-SNE (PCA)", fontsize=15)
r2_plot2 = plt.scatter(pca["x1"], pca["x2"], c=r2.values, cmap="viridis", alpha=0.6)

# %% [markdown]
# ## 3: K-Means clustering after PCA

# %%
r3 = pd.DataFrame(kmeans.fit_predict(pca), columns=["cluster"])
r3.set_index(data.index, inplace=True)

plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
plt.title("K-Means clustering after PCA (t-SNE)", fontsize=15)
r3_plot1 = plt.scatter(tsne["x1"], tsne["x2"], c=r3.values, cmap="viridis", alpha=0.6)

plt.subplot(1, 2, 2)
plt.title("K-Means clustering after PCA (PCA)", fontsize=15)
r3_plot2 = plt.scatter(pca["x1"], pca["x2"], c=r3.values, cmap="viridis", alpha=0.6)

# %% [markdown]
# ## Select reduced demension target for PCA

# %%
# 95% 분산 유지를 위한 최소한의 차원 수를 계산하여 PCA 모델에 적용
model_pca_opt = PCA(n_components=0.95)
pca_opt = model_pca_opt.fit_transform(data)
print(f"Number of demension: {model_pca_opt.n_components_}")

# %% [markdown]
# ## 4: K-Means Clustering for optimized PCA

# %%
r4 = pd.DataFrame(kmeans.fit_predict(pca_opt), columns=["cluster"])
r4.set_index(data.index, inplace=True)

plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
plt.title("K-Means clustering after PCA (t-SNE)", fontsize=15)
r4_plot1 = plt.scatter(tsne["x1"], tsne["x2"], c=r4.values, cmap="viridis", alpha=0.6)

plt.subplot(1, 2, 2)
plt.title("K-Means clustering after PCA (PCA)", fontsize=15)
r4_plot2 = plt.scatter(pca["x1"], pca["x2"], c=r4.values, cmap="viridis", alpha=0.6)


# %%
# run t-SNE
r4_data = data.copy()
r4_data["clusters"] = r4.values
r4_tsne = TSNE(n_components=n_components, learning_rate=300).fit_transform(r4_data)

plt.title("t-SNE on K-Means clustering", fontsize=15)
plt.scatter(r4_tsne[:, 0], r4_tsne[:, 1], c=r4.values, cmap="viridis", alpha=0.6)
plt.colorbar()
plt.show()

# %% [markdown]
# ## Mimax Scaling

# %%
data.head()


# %%
# data_clean=data.copy()
# data_clean.drop(["chroma_cqt", "chroma_stft"], axis=1, inplace=True)
# data_clean.drop("kurtosis", axis=1, level=1)

# Min-max scaling
from sklearn.preprocessing import MinMaxScaler

scaler_minmax = MinMaxScaler(feature_range=(0, 1))
data_minmax = pd.DataFrame(data)
data_minmax.iloc[:, :] = scaler_minmax.fit_transform(data)
data_minmax.head()


# %%

# Standardization
from sklearn.preprocessing import StandardScaler

scaler_standard = StandardScaler()
data_standard = pd.DataFrame(data)
data_standard.iloc[:, :] = scaler_standard.fit_transform(data)
data_standard.head()

# %% [markdown]
# Min-Max Scaling을 수행한 데이터에 대해 차원 축소 수행

# %%
# t-sne 모델 생성 및 수행
from sklearn.manifold import TSNE

model_tsne = TSNE(n_components=2, learning_rate=300)
time_start = time.time()
tsne_minmax = pd.DataFrame(model_tsne.fit_transform(data_minmax), columns=["x1", "x2"])
tsne_minmax.set_index(data_minmax.index, inplace=True)
print(f"t-SNE Done. Elepsed Time:{time.time()-time_start}")

# PCA 수행
from sklearn.decomposition import PCA

model_pca = PCA(n_components=2)
time_start = time.time()
pca_minmax = pd.DataFrame(model_pca.fit_transform(data_minmax), columns=["x1", "x2"])
pca_minmax.set_index(data_minmax.index, inplace=True)
print(f"PCA Done. Elepsed Time:{time.time()-time_start}")

# %% [markdown]
# ## PCA and t-SNE for minmax data

# %%
# t-SNE와 PCA 수행 결과 2차원 공간에 출력
plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
plt.title("t-SNE for minmax data", fontsize=15)
r0_plot_tsne = plt.scatter(
    tsne_minmax["x1"],
    tsne_minmax["x2"],
    c=range(0, tsne.shape[0]),
    cmap="viridis",
    alpha=0.6,
)

plt.subplot(1, 2, 2)
plt.title("PCA for minmax data", fontsize=15)
r0_plot_pca = plt.scatter(
    pca_minmax["x1"],
    pca_minmax["x2"],
    c=range(0, pca.shape[0]),
    cmap="viridis",
    alpha=0.6,
)

# %% [markdown]
# ## 5: K-Means with PCA, minmax scaling

# %%
r5 = pd.DataFrame(kmeans.fit_predict(pca_minmax), columns=["cluster"])
r5.set_index(data.index, inplace=True)

plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
plt.title("K-Means clustering after PCA (t-SNE)", fontsize=15)
r5_plot1 = plt.scatter(
    tsne_minmax["x1"], tsne_minmax["x2"], c=r5.values, cmap="viridis", alpha=0.6
)

plt.subplot(1, 2, 2)
plt.title("K-Means clustering after PCA (PCA)", fontsize=15)
r5_plot2 = plt.scatter(
    pca_minmax["x1"], pca_minmax["x2"], c=r5.values, cmap="viridis", alpha=0.6
)

# %% [markdown]
# ## 6: K-Means with optimized PCA, minmax scaling

# %%
model_pca_opt = PCA(n_components=0.95)
pca_opt_minmax = model_pca_opt.fit_transform(data_minmax)
print(f"Number of demension: {model_pca_opt.n_components_}")

r6 = pd.DataFrame(kmeans.fit_predict(pca_opt_minmax), columns=["cluster"])
r6.set_index(data.index, inplace=True)

plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
plt.title("K-Means clustering after PCA (t-SNE), minmax-scaled", fontsize=15)
r6_plot1 = plt.scatter(
    tsne_minmax["x1"], tsne_minmax["x2"], c=r6.values, cmap="viridis", alpha=0.6
)

plt.subplot(1, 2, 2)
plt.title("K-Means clustering after PCA (PCA), minmax-scaled", fontsize=15)
r6_plot2 = plt.scatter(
    pca_minmax["x1"], pca_minmax["x2"], c=r6.values, cmap="viridis", alpha=0.6
)

# %% [markdown]
# ## 7: K-Means with t-SNE, minmax scaling

# %%
r7 = pd.DataFrame(kmeans.fit_predict(tsne_minmax), columns=["cluster"])
r7.set_index(data.index, inplace=True)

plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
plt.title("K-Means clustering after t-SNE (t-SNE), minmax-scaled", fontsize=15)
r7_plot1 = plt.scatter(
    tsne_minmax["x1"], tsne_minmax["x2"], c=r7.values, cmap="viridis", alpha=0.6
)

plt.subplot(1, 2, 2)
plt.title("K-Means clustering after t-SNE (PCA), minmax-scaled", fontsize=15)
r7_plot2 = plt.scatter(
    pca_minmax["x1"], pca_minmax["x2"], c=r7.values, cmap="viridis", alpha=0.6
)

# %% [markdown]
# ## 8: K-Means with t-SNE, minmax scaling, 50 clusters

# %%
r8 = pd.DataFrame(kmeans_50.fit_predict(tsne_minmax), columns=["cluster"])
r8.set_index(data.index, inplace=True)

plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
plt.title("K-Means clustering after t-SNE (t-SNE), minmax-scaled", fontsize=15)
r8_plot1 = plt.scatter(
    tsne_minmax["x1"], tsne_minmax["x2"], c=r8.values, cmap="viridis", alpha=0.6
)

plt.subplot(1, 2, 2)
plt.title("K-Means clustering after t-SNE (PCA), minmax-scaled", fontsize=15)
r8_plot2 = plt.scatter(
    pca_minmax["x1"], pca_minmax["x2"], c=r8.values, cmap="viridis", alpha=0.6
)
