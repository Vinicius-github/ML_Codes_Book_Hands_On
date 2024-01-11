#%%
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from skimage.io import imshow

#%%
data = fetch_olivetti_faces()
print(data.DESCR)

# %%
X = data.images
Y = data.target
# %%
imshow(X[20]) 

#%%
Counter(Y)

# %%
x, x_test, y, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
# %%
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.80)

#%%

plt.plot(*zip(*sorted(Counter(y_test).items())))

# %%
pd.DataFrame(Counter(y_test).items(), columns=['Key', 'Value']).sort_values(by=['Key'])
# %%
Z = data.images
W = data.target
splitter = StratifiedShuffleSplit(n_splits=1, 
                                  test_size=0.10, 
                                  random_state=42)
for train_index, test_index in splitter.split(Z,W):
    z_train, w_train = Z[train_index], W[train_index]
    z_test, w_test = Z[test_index], W[test_index]

# %%
pd.DataFrame(Counter(w_test).items(), columns=['Key', 'Value']).sort_values(by=['Key'])
# %%
len(z_test)
# %%
