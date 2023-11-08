import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
ind = dataset.iloc[:,1:-1].values
dep = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(ind,dep)

pred = regressor.predict([[6.5]])
print(pred)

#Visualisation of the Random Forest Model
xgrid = np.arange(min(ind),max(ind),0.1)
xgrid = xgrid.reshape(len(xgrid),1)
plt.scatter(ind,dep,color = 'red')
plt.plot(xgrid,regressor.predict(xgrid),color = 'blue')
plt.title('Random Forest Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()