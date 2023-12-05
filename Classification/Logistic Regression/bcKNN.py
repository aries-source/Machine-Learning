# K - Nearest Neighbor

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

# Scaling the features
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
indTrain = scale.fit_transform(indTrain)
indTest = scale.transform(indTest)

# Training and fitting the KNN model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors= 5,metric= 'minkowski',p=2)
classifier.fit(indTrain,depTrain)

# Predicting a single observation
# person = classifier.predict(scale.transform([[30,87000]]))
# print(person)
# person = classifier.predict(indTest[0,0:2].reshape(1,-1))
# print(person)

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