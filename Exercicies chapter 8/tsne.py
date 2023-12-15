
#Load libraries
import matplotlib as mpl
from sklearn.datasets import fetch_openml
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

#Load dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print('dados carregados')

#Split data
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)

#Reduce dimension
tsne = TSNE(n_components=2, 
            random_state=42, 
            n_jobs=-1)

x_reduced = tsne.fit_transform(x_train)

#Plot values
#Create Datframe to PC1, PC2 and Target (y_train)
tsne_df = pd.DataFrame(
np.column_stack((x_reduced[:,0], x_reduced[:,1], y_train)),
columns=["pc1", "pc2", "y"])
#only sort values
tsne_df.loc[:, "y"] = tsne_df.y.astype(int) 
#plot figure
grid = sns.FacetGrid(tsne_df, hue="y", height=6)
grid.map(plt.scatter, 'pc1', 'pc2').add_legend()