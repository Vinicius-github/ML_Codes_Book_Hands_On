#%%
#Load dataset MNIST from sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn import svm

#Load MNIST
mnist = fetch_openml('mnist_784', version=1, parser='auto')
print('dados carregados')

#Create train and test dataset
X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)
print('dados separados')
#Create diferrent algorithms
rfclf = RandomForestClassifier(random_state=42, n_jobs=-1)
extclf = ExtraTreesClassifier(random_state=42)
svmclf = svm.SVC(random_state=42)
print('modelos construidos')
#Train models 
estimators = [rfclf, extclf, svmclf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)
#Print the results for each model
print([estimator.score(X_val, y_val) for estimator in estimators])
#Run the Voting classifier hard with the algorithms created before
vtclf_hard = VotingClassifier(estimators=[('rf', rfclf), 
                                     ('ext', extclf), 
                                     ('svm', svmclf)],
                                     n_jobs=-1,
                                     verbose=2, 
                                     voting='hard')
# Train models
vtclf_hard.fit(X_train, y_train)
print('modelo treinado')
#Print the score the models
print(vtclf_hard.score(X_val, y_val))
#Run the Voting classifier soft with the algorithms created before
vtclf_soft = VotingClassifier(estimators=[('rf', rfclf), 
                                     ('ext', extclf)],
                                     n_jobs=-1,
                                     verbose=1,
                                     voting='soft')
vtclf_soft.fit(X_train, y_train)
print(vtclf_soft.score(X_val, y_val))