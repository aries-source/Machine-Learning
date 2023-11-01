import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset 
dataset = pd.read_csv('50_Startups.csv')
indVariables = dataset.iloc[:,:-1].values
depVariable = dataset.iloc[:,-1].values

# Encoding Categorical Variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

Encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')
indVariables=np.array(Encoder.fit_transform(indVariables))

print(indVariables)

from sklearn.model_selection import train_test_split
indTrain, indTest, depTrain, depTest = train_test_split(indVariables,depVariable,test_size= 0.2, random_state=0)

# Model Fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(indTrain,depTrain)
depPred = regressor.predict(indTest)

print(np.concatenate((depPred.reshape(len(depPred),1),depTest.reshape(len(depPred),1)),1))

# Predicting a profit
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# Getting the Coefficients
print(regressor.coef_)
print(regressor.intercept_)

