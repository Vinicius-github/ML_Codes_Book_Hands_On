#Load dataset MNIST from sklearn
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import datetime 

#Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print('dados carregados')

x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)

start_time = datetime.datetime.now()
#Train model without PCA
rfclf = RandomForestClassifier(n_estimators=100, 
                               random_state=42, 
                               n_jobs=-1)
rfclf.fit(x_train, y_train)
print(datetime.datetime.now() - start_time)

#Execute PCA with 95% variance
pca_cl = PCA(n_components=0.95)
X_train_reduced = pca_cl.fit_transform(x_train) #Fit the model with X and apply the dimensionality reduction on X.

start_time_pca = datetime.datetime.now()
#Train model with PCA
rfclf_pca = RandomForestClassifier(n_estimators=100, 
                                   random_state=42, 
                                   n_jobs=-1)
rfclf_pca.fit(X_train_reduced, y_train)
print(datetime.datetime.now() - start_time_pca)

#Random forest predctions without PCA and with PCA
predict_rfclf = rfclf.predict(x_test)
print("A acurácia do modelo de Random Forest sem PCA é: ", accuracy_score(y_test, predict_rfclf))

x_test_reduced = pca_cl.transform(x_test) #Apply dimensionality reduction to X.
predict_rfclf_pca = rfclf_pca.predict(x_test_reduced) 
print("A acurácia do modelo de Random Forest com PCA é: ", accuracy_score(y_test, predict_rfclf_pca))



