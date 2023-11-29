#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

#Load dataset MNIST from sklearn
mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)

# print(mnist.keys())
X, y = mnist["data"], mnist["target"]

#Create train and test dataset
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Create a classifier: a support vector classifier

# lin_clf = LinearSVC(random_state=42)
# lin_clf.fit(X_train, y_train)

clf = svm.SVC()

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted_train = clf.predict(X_train)
predicted = clf.predict(X_test)

# Visualize the first 4 test samples and show their predicted digit value in the title
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# Results
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

#Accuracy
print("Accuracy with train values: ", accuracy_score(y_train, predicted_train))
print("Accuracy with test values: ", accuracy_score(y_test, predicted))


