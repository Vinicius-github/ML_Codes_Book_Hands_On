#%%
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from skimage.io import imshow

# Load dataset
data = fetch_olivetti_faces()
print(data.DESCR)

# Separando os dados da imagem e target
X = data.images
Y = data.target

# Mostrando uma imagem
imshow(X[20]) 

# Contabilizando a quantidade de dados do target
Counter(Y)

# SPlit dataset
# Aqui foi utilizado o Stratified para garantir que as figuras aparececem em todos os casos
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
train_valid_idx, test_idx = next(strat_split.split(X, Y))
X_train_valid = X[train_valid_idx]
y_train_valid = Y[train_valid_idx]
X_test = X[test_idx]
y_test = Y[test_idx]

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]

# Dimensionalidade após o split
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

# Para conseguir rodar o Kmeans foi precisso fazer o reshape dos dados, coloca-lo em 2D.
# https://awari.com.br/reshape-em-python-aprenda-a-transformar-dados-de-forma-eficiente/
X_train = X_train.reshape(-1, 4096)

# Criando o modelo do K_means
k_range = range(5, 150, 1)
kmeans_per_k = []
for k in k_range:
    print("k={}".format(k))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train)
    kmeans_per_k.append(kmeans)

# Utilizando a validação da silhouete
silhouette_scores = [silhouette_score(X_train, model.labels_)
                     for model in kmeans_per_k]
best_index = np.argmax(silhouette_scores)
best_k = k_range[best_index]
best_score = silhouette_scores[best_index]

plt.figure(figsize=(8, 3))
plt.plot(k_range, silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.plot(best_k, best_score, "rs")
plt.show()
print(best_k)

# Utilizando a validaçção do Elbow (Inertia)
inertias = [model.inertia_ for model in kmeans_per_k]
best_inertia = inertias[best_index]

plt.figure(figsize=(8, 3.5))
plt.plot(k_range, inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.plot(best_k, best_inertia, "rs")
plt.show()
best_model = kmeans_per_k[best_index]
print(best_model)

# Em ambos os casos a quantidade de Custer boa para o conjunto de dados inicialmente foi 100.

# Mostrando os resultados de cada um dos 94 Clusters.
def plot_faces(faces, labels, n_cols=5):
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face, cmap="gray")
        plt.axis("off")
        plt.title(label)
    plt.show()

for cluster_id in np.unique(best_model.labels_):
    print("Cluster", cluster_id)
    in_cluster = best_model.labels_==cluster_id
    faces = X_train[in_cluster]
    labels = y_train[in_cluster]
    plot_faces(faces, labels)
