# Support Vector Machines with a Linear Kernel

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
ind = dataset.iloc[:,:-1].values
dep = dataset.iloc[:,-1].values

# Splitting Test and Train sets
from sklearn.model_selection import train_test_split
indTrain,indTest,depTrain,depTest = train_test_split(ind,dep,test_size=0.25,random_state=0)   

# Scaling the features
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
indTrain = scale.fit_transform(indTrain)
indTest = scale.transform(indTest)

# Train and fitting the logistic regression model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(indTrain,depTrain)

# Predicting a single observation
person = classifier.predict(scale.transform([[30,87000]]))
print(person)
person = classifier.predict(indTest[0,0:2].reshape(1,-1))
print(person)

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


# Visualising the train set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = scale.inverse_transform(indTrain), depTrain
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
#                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# plt.contourf(X1, X2, classifier.predict(scale.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
# plt.title('Support Vector Machine (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()


# Visualising the test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = scale.inverse_transform(indTest), depTest
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
#                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# plt.contourf(X1, X2, classifier.predict(scale.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
# plt.title('Support Vector Machines (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()


