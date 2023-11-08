import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
ind = dataset.iloc[:,1:-1].values
dep = dataset.iloc[:,-1].values

#StandardScaler expects a 2D array hence dep needs to be reshaped
dep = dep.reshape(len(dep),1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
ScaleX = StandardScaler()
ScaleY = StandardScaler()
ind = ScaleX.fit_transform(ind)

dep = ScaleY.fit_transform(dep)
print(ind)
print(dep)

#SVR Model
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(ind,dep)

#Predicting scaled 6.5
depPred = regressor.predict(ScaleX.transform([[6.5]]))
predSal = ScaleY.inverse_transform(depPred.reshape(-1,1))
print(predSal)

#Visualisation
plt.scatter(ScaleX.inverse_transform(ind),ScaleY.inverse_transform(dep),color = 'red')
plt.plot(ScaleX.inverse_transform(ind),ScaleY.inverse_transform(regressor.predict(ind).reshape(-1,1)), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#High Resolution
xgrid = np.arange(min(ScaleX.inverse_transform(ind)),max(ScaleX.inverse_transform(ind)),0.1)
xgrid = xgrid.reshape(len(xgrid),1)
plt.scatter(ScaleX.inverse_transform(ind),ScaleY.inverse_transform(dep),color = 'red')
plt.plot(xgrid,ScaleY.inverse_transform(regressor.predict(ScaleX.transform(xgrid)).reshape(-1,1)), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

