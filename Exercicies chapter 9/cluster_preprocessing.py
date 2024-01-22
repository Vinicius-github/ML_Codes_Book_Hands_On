#%%
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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
imshow(X[50]) 

print(np.shape(X))

# %%
# Train teste aplit
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
# %%
# Duas maneiras de fazer o reshape dos dados
# Se colocar o -1 na frente ele irá manter a quantidade de linhas dos dados originais
reshape_X_train = X_train.reshape(-1,64*64)
print(np.shape(reshape_X_train))

reshape_X_valid = X_valid.reshape(-1,64*64)
print(np.shape(reshape_X_valid))

# Se colocar o -1 no segundo local ele materá a quantidade de colunas originais 64*64=4096
# reshape_X_train = X_train.reshape(len(X_train),-1)
# print(np.shape(reshape_X_train))

# %%
# Treinamento do modelo sem o Kmeans
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(reshape_X_train, y_train)
    
print(clf.score(reshape_X_valid, y_valid))


# %%
# Processo usando o Pipeline considerando o Kmeans como preprocessamento
k_range = range(5, 150, 1)
clusters = []
kmeans_per_k = []
for n_clusters in k_range:
    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42)),
        ("forest_clf", RandomForestClassifier(n_estimators=150, random_state=42))
    ])
    pipeline.fit(reshape_X_train, y_train)
    clusters.append(n_clusters)
    kmeans_per_k.append(pipeline.score(reshape_X_valid, y_valid))
    print(n_clusters, pipeline.score(reshape_X_valid, y_valid))
# %%
#Score máximo encontrado utilizando o préprocessamento com clusterização
max_value = max(kmeans_per_k)
# kmeans_per_k.index(max_value)
# Número de clusters
# clusters[73]
# Neste teste podemos que o préprocessamento com a custerização 
# não obteve e melhores resultados 0.93 contra 0.85