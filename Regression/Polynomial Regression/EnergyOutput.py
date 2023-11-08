import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =  pd.read_csv('Data.csv')
ind = dataset.iloc[:,:-1].values
dep = dataset.iloc[:,-1].values

# Splitting into Test and Train
from sklearn.model_selection import train_test_split
indTrain, indTest, depTrain, depTest = train_test_split(ind,dep, test_size=0.2, random_state=0)


# Polynomial model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

Degree = PolynomialFeatures(degree=2)
indPoly = Degree.fit_transform(indTrain)
regressorPoly = LinearRegression()
regressorPoly.fit(indPoly,depTrain)


# Predicting
polPred = regressorPoly.predict(Degree.transform(indTest))
np.set_printoptions(precision=2)
print(np.concatenate((polPred.reshape(len(polPred),1), depTest.reshape(len(depTest),1)),1))

# Accuracy 
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,r2_score
mae = mean_absolute_error(depTest,polPred)
mse = mean_squared_error(depTest,polPred)
mape = mean_absolute_percentage_error(depTest,polPred)
R_square = r2_score(depTest,polPred)

print(f'Mean Absolute Error is {mae}')
print(f'Mean Squared Error is {mse}')
print(f'Mean Absolute Percentage Error is {mape}')
print(f'R-Square is {R_square}')


