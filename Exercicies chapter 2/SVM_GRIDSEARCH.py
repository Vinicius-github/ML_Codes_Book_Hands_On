#import libraries
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

#load dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
dataset = pd.read_csv(url)

#Split dataset with Stratified
dataset['income_cat'] = pd.cut(dataset["median_income"],
        bins = [0.,1.5,3.0,4.5,6.,np.inf])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income_cat"]):
    strat_train_set=dataset.loc[train_index]
    strat_test_set=dataset.loc[test_index]
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)

print(strat_train_set.size)
print(strat_test_set.size)

#Model pipeline

#Test data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, list(housing.select_dtypes(include="float64"))),
    ("cat", OneHotEncoder(handle_unknown = 'ignore'), list(housing.select_dtypes(include="object")))
])

model_pipeline = Pipeline([
    ("preprocess", full_pipeline),
    ("svr", SVR())
])

param_grid = [
        {'svr__kernel': ['linear'], 'svr__C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'svr__kernel': ['rbf'], 'svr__C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

grid_svr = GridSearchCV(model_pipeline, 
                        param_grid, 
                        cv=5, 
                        scoring='neg_mean_squared_error', 
                        verbose=2)

final_model = grid_svr.fit(housing,housing_labels)

#best score
negative_mse = final_model.best_score_
rmse = np.sqrt(-negative_mse)
rmse

#best parameters
final_model.best_params_
