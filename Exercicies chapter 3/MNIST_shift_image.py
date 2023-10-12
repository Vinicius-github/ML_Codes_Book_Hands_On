#load libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#Load dataset MNIST from sklearn
#Necessary put as_frame=false to dataset comming like series
mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
# print(mnist.keys())
X, y = mnist["data"], mnist["target"]

#Print first numeric value
print("First image dataset \n")
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()


#Create shift def
def shift_digit(digit_array, dx, dy):
    return shift(digit_array.reshape(28, 28), [dy, dx]).reshape(784)

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

#Print some imagens shifted
#Shift Left image
print("Shift image left\n")
plot_digit(shift_digit(some_digit, -5, 0))

#Shift top image
print("Shift image top\n")
plot_digit(shift_digit(some_digit, 0, -5))


#Create train and test dataset
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#Add shifted image to train dataset
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

#Create model, train and test
knn_clf = KNeighborsClassifier()

param_grid = [{ 
    "weights" : ['uniform', 'distance'],
    "n_neighbors" : [1,3,5,7,10]
}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
grid_search.fit(X_train_augmented, y_train_augmented)

print(grid_search.best_params_)
print(grid_search.best_score_)

y_pred = grid_search.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_pred))

