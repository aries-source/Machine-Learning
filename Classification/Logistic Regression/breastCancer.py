# Logistic Regression

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('breast_cancer.csv')
ind = dataset.iloc[:,1:-1].values
dep = dataset.iloc[:,-1].values

# Splitting Test and Train sets
from sklearn.model_selection import train_test_split
indTrain,indTest,depTrain,depTest = train_test_split(ind,dep,test_size=0.2,random_state=0)

# Train and fitting the logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(indTrain,depTrain)

# Predicting The Entire test set
purchase = classifier.predict(indTest)
np.set_printoptions(precision=2)
print(np.concatenate((depTest.reshape(len(depTest),1),purchase.reshape(len(depTest),1)),1))

# Creating the Confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
matrix = confusion_matrix(depTest,purchase)
accuracy = accuracy_score(depTest,purchase)
print(matrix)
print(accuracy)


# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
cV = cross_val_score(estimator=classifier,X=indTrain,y=depTrain, cv=10)
print('Accuracy : {:.2f} %'.format(cV.mean()*100))
print('Standard Deviation : {:.2f} %'.format(cV.std()*100))
