import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
ind = dataset.iloc[:,1:-1].values
dep = dataset.iloc[:,-1].values

#dep = dep.reshape(len(dep),1)

#Training The Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(ind,dep)

depPred = regressor.predict([[6.5]])
print(depPred)

# Visualisation of the decision tree model in high resolution
xgrid = np.arange(min(ind),max(ind),0.1)
xgrid = xgrid.reshape(len(xgrid),1)
plt.scatter(ind,dep,color = 'red')
plt.plot(xgrid,regressor.predict(xgrid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
