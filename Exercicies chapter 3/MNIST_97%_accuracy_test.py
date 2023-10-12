#%%
#load libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier



#Load dataset MNIST from sklearn
mnist = fetch_openml('mnist_784', version=1, parser='auto')
# print(mnist.keys())
X, y = mnist["data"], mnist["target"]
# print(X.head(1))
#Cada linha representa a combinação de pixels para formar o número. Os pixel varia entre 0 e 255

#Create train and test dataset
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Test options and evaluation metric
num_folds = 5
seed = 86
scoring = 'accuracy'

# Spot-Check Algorithms
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(("SGD", SGDClassifier()))

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=num_folds, 
                  random_state=seed, 
                  shuffle=True)
    
    cv_results = cross_val_score(model, 
                                 X_train, 
                                 y_train, 
                                 cv=kfold, 
                                 scoring=scoring, 
                                 verbose=2)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Evaluate Algorithms: Standardize Data
# Standardize the dataset
pipelines = []
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSGD', Pipeline([('Scaler', StandardScaler()),('SVM', SGDClassifier())])))

results = []
names = []

for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, 
                  random_state=seed, 
                  shuffle=True)
    
    cv_results = cross_val_score(model, 
                                 X_train, 
                                 y_train, 
                                 cv=kfold, 
                                 scoring=scoring,
                                 verbose=2)
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# %%
# Of the algorithms tested, KNN without scalar treatment obtained the best 
# result, reaching 0.97%. However, a Gridsearch will be carried out in an 
# attempt to improve the final result.

from sklearn.metrics import accuracy_score

param_grid = [{ 
    "weights" : ['uniform', 'distance'],
    "n_neighbors" : [1,3,5,7,10]
}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=2)
grid_search.fit(X_train, y_train)

#%%
print(grid_search.best_params_)
print(grid_search.best_score_)


y_pred = grid_search.best_estimator_.predict(X_test)
accuracy_score(y_test, y_pred)