import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =  pd.read_csv('Position_Salaries.csv')
ind = dataset.iloc[:,1:-1].values
dep = dataset.iloc[:,-1].values

# Due to the limited data points we will not split the dataset
# from sklearn.model_selection import train_test_split
# indTrain, indTest, depTrain, depTest = train_test_split(ind,dep, test_size=0.2, random_state=0)

# Fitting a simple linear model for comparison
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(ind,dep)

# Polynomial model
from sklearn.preprocessing import PolynomialFeatures
Degree = PolynomialFeatures(degree=4)
indPoly = Degree.fit_transform(ind)
regressorPoly = LinearRegression()
regressorPoly.fit(indPoly,dep)

# Visualizing the simple linear model
plt.scatter(ind,dep,color = 'red')
plt.plot(ind, regressor.predict(ind),color = 'blue')
plt.title('Simple Linear Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the polynomial linear model
plt.scatter(ind,dep,color = 'red')
plt.plot(ind, regressorPoly.predict(indPoly),color = 'blue')
plt.title('Polynomial Linear Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# For a smother curve
xgrid = np.arange(min(ind),max(ind),0.1)
xgrid = xgrid.reshape(len(xgrid),1)
plt.scatter(ind,dep,color = 'red')
plt.plot(xgrid, regressorPoly.predict(Degree.fit_transform(xgrid)),color = 'blue')
plt.title('Polynomial Linear Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting the salary of position level 6.5 
Lin=regressor.predict([[6.5]])
Pol = regressorPoly.predict(Degree.fit_transform([[6.5]]))
print(f'The predicted salary for an individual with 6.5 level from simple linear model is {Lin} ')
print(f'The predicted salary for an individual with 6.5 level from polynomial linear model is {Pol} ')
