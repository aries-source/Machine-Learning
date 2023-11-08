# Support Vector Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
ind = dataset.iloc[:,:-1].values
dep = dataset.iloc[:,-1].values

#StandardScaler expects a 2D array hence dep needs to be reshaped
dep = dep.reshape(len(dep),1)

# Splitting into Test and Train
from sklearn.model_selection import train_test_split
indTrain,indTest,depTrain,depTest = train_test_split(ind,dep,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
ScaleX = StandardScaler()
ScaleY = StandardScaler()

indTrain = ScaleX.fit_transform(indTrain)
depTrain = ScaleY.fit_transform(depTrain)


#SVR Model
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(indTrain,depTrain)

#Predicting Test set
depPred = regressor.predict(ScaleX.transform(indTest))
predEnergy = ScaleY.inverse_transform(depPred.reshape(-1,1))

# Printing Predicted and Test
np.set_printoptions(precision=2)
print(np.concatenate((depTest.reshape(len(depTest),1),predEnergy.reshape(len(depTest),1)),1))

# Accuracy
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,r2_score
mae = mean_absolute_error(depTest,predEnergy)
mse = mean_squared_error(depTest,predEnergy)
mape = mean_absolute_percentage_error(depTest,predEnergy)
R_square = r2_score(depTest,predEnergy)

print(f'Mean Absolute Error is {mae}')
print(f'Mean Squared Error is {mse}')
print(f'Mean Absolute Percentage Error is {mape}')
print(f'R-Square is {R_square}')
