import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Fetching dataset
housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]

#Split daaset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train
svr = SVR()
svr.fit(X_train_scaled, y_train)

#Predict
y_pred = svr.predict(X_train_scaled)

#Results train
mse = mean_squared_error(y_train, y_pred)
print("The mse value for train dataset is: ", mse)
print("The RMSE value for train dataset is: ", np.sqrt(mse))

#Predict
y_pred_test = svr.predict(X_test_scaled)

#Results test
mse_test = mean_squared_error(y_test, y_pred_test)
print("The mse value for train dataset is: ", mse_test)
print("The RMSE value for train dataset is: ", np.sqrt(mse_test))

#Ploty the top 10 values train dataset
plt.plot(y_train[:10], color = "blue")
plt.plot(y_pred[:10], color = "red")
plt.title('SVR - Top ten values (Train dataset)')
plt.xlabel('Position level')
plt.ylabel('Housing value')
plt.show()

#Ploty the top 10 values train dataset
plt.plot(y_test[:10], color = "blue")
plt.plot(y_pred_test[:10], color = "red")
plt.title('SVR - Top ten values (Test dataset)')
plt.xlabel('Position level')
plt.ylabel('Housing value')
plt.show()

