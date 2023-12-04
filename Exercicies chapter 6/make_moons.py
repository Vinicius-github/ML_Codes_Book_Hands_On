#%%
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


#Create a random sample make_moons
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Create grid parameters for hyperparameter tuning
params =  {
    'min_samples_leaf': [1, 2, 3, 4, 5, 6],
    'max_leaf_nodes': [1, 2, 3, 4, 5, 6]
}

# Create gridsearch instance
grid = GridSearchCV(estimator=clf,
                    param_grid=params,
                    cv=5,
                    n_jobs=1,
                    verbose=1)

# Fit the model
grid.fit(X_train, y_train)

# Assess the score
grid.best_score_, grid.best_params_
# %%
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
# %%
