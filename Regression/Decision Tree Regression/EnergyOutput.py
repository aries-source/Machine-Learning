# Decision Tree Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
ind = dataset.iloc[:,:-1].values
dep = dataset.iloc[:,-1].values

# Splitting into Test and Train
from sklearn.model_selection import train_test_split
indTrain, indTest,depTrain,depTest = train_test_split(ind,dep,test_size=0.2,random_state=0)

#Training The Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(indTrain,depTrain)

# Predicting the test set
depPred = regressor.predict(indTest)
np.set_printoptions(precision=2)
print(np.concatenate((depTest.reshape(len(depTest),1),depPred.reshape(len(depTest),1)),1))

# Accuracy
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,r2_score
mae = mean_absolute_error(depTest,depPred)
mse = mean_squared_error(depTest,depPred)
mape = mean_absolute_percentage_error(depTest,depPred)
R_square = r2_score(depTest,depPred)

print(f'Mean Absolute Error is {mae}')
print(f'Mean Squared Error is {mse}')
print(f'Mean Absolute Percentage Error is {mape}')
print(f'R-Square is {R_square}')