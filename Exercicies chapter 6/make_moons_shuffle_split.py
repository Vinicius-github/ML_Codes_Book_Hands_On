#%%
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Create dataset
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

#Split with shuffle
X_train, X_teste, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a shuffle
n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

#Create Forest Classifier
forest = [DecisionTreeClassifier(max_leaf_nodes=4, min_samples_leaf=1,random_state=42) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    
    y_pred = tree.predict(X_teste)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

#The mean results create by forest
np.mean(accuracy_scores)

#Create predictions for each test instance and maintain the highest frequency
Y_pred = np.empty([n_trees, len(X_teste)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_teste)

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

#Print result
accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))

# %%
