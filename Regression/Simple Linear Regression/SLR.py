#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import and defining dataset
dataset = pd.read_csv('salary_Data.csv')
indVariable = dataset.iloc[:,:-1].values
depVariable = dataset.iloc[:,-1].values

# Splitting into Train and Test sets
from sklearn.model_selection import train_test_split
indVariableTrain,indVariableTest,depVariableTrain,depVariableTest = train_test_split(indVariable,depVariable,test_size=0.2,random_state=0)

# Fitting the training sets
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(indVariableTrain,depVariableTrain)

# Predicting the test sets
depPred = regressor.predict(indVariableTest)

# Visualizing the train set
plt.scatter(indVariableTrain,depVariableTrain,color = 'red')
plt.plot(indVariableTrain,regressor.predict(indVariableTrain), color= 'blue')
plt.title('Visualizing the train set') 
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the test set
plt.scatter(indVariableTest,depVariableTest,color = 'red')
plt.plot(indVariableTest,depPred, color= 'blue')
# plt.plot(indVariableTrain,regressor.predict(indVariableTrain), color= 'blue')
plt.title('Visualizing the test set') 
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Accuracy
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,r2_score
mae = mean_absolute_error(depVariableTest,depPred)
mse = mean_squared_error(depVariableTest,depPred)
mape = mean_absolute_percentage_error(depVariableTest,depPred)
R_square = r2_score(depVariableTest,depPred)

print(f'Mean Absolute Error is {mae}')
print(f'Mean Squared Error is {mse}')
print(f'Mean Absolute Percentage Error is {mape}')
print(f'R-Square is {R_square}')

# Predicting a single input
print(regressor.predict([[12]]))

# Retrieving Parameters of the model
print(regressor.coef_)
print(regressor.intercept_)

