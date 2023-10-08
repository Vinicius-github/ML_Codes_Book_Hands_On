#%%
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
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
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

#Create pipelines

#Transforms
num_preprocess = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_preprocess = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

pipe_preprocess = ColumnTransformer([
    ('num',num_preprocess,list(housing.select_dtypes(include="float64"))),
    ('cat',cat_preprocess,list(housing.select_dtypes(include="object")))
])

model_rfreg = Pipeline([
    ("preprocess", pipe_preprocess),
    ('rfreg', RandomForestRegressor())
])

model_svr = Pipeline([
    ("preprocess", pipe_preprocess),
    ('svr', SVR())
])

#Configure RandomizedSearchCV 
param_grid_svr = [
        {"preprocess__num__imputer__strategy": ["mean", "median", "most_frequent"],
        'svr__kernel': ['linear', 'rbf'], 
        'svr__C': [1.0, 3.0, 10., 30., 100., 300., 1000., 3000., 10000., 30000.0],
        'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}
    ]


print("ok")

#Run RandomizedSearchCV and fit model
rnd_svr = RandomizedSearchCV(model_svr, 
                        param_grid_svr, 
                        cv=5, 
                        scoring='neg_mean_squared_error', 
                        verbose=2,
                        random_state=86)

#fit model
svr_model = rnd_svr.fit(housing,housing_labels)

#Best estimator
svr_model.best_estimator_

#Best parameters
svr_model.best_params_

#Best score
svr_model.best_score_

# Compile top 5
cv_results = pd.DataFrame(svr_model.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)
cv_results[
    [
        "mean_test_score",
        "std_test_score",
        "param_preprocess__num__imputer__strategy",
        "param_svr__kernel",
        "param_svr__C",
        "param_svr__gamma"
    ]
].head(5)
# %%
