#import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR, LinearSVR

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

#Train data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#Test data
housing_test = strat_test_set.drop("median_house_value", axis=1)
housing_labels_test = strat_test_set["median_house_value"].copy()

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

model_svr = Pipeline([
    ("preprocess", pipe_preprocess),
    ('svr', SVR())
])

model_linear_svr = Pipeline([
    ("preprocess", pipe_preprocess),
    ('linear_svr', LinearSVR())
])

model_svr.fit(housing, housing_labels)
model_linear_svr.fit(housing, housing_labels)

y_pred_train = model_svr.predict(housing)
mse_train = mean_squared_error(housing_labels, y_pred_train)
print(mse_train)
print(np.sqrt(mse_train))
