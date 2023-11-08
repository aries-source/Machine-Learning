# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset

Data = pd.read_csv("Data.csv")
print(Data.head(5))

#Extracting Independent and Dependent Variables

independentVariable = Data.iloc[:,:-1].values
dependentVariable = Data.iloc[:,-1].values

#Dealing With Missing Values
# Identify missing data (assumes that missing data is represented as NaN)
missing = Data.isna()
count = missing.sum()
# Print the number of missing entries in each column
print(count)
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values= np.nan, strategy= 'mean')
impute.fit(independentVariable[:,1:3])
independentVariable[:,1:3]=impute.transform(independentVariable[:,1:3])
# print(independentVariable)

# Encoding Categorical Data
# Independent Variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
objectClass = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
independentVariable = np.array(objectClass.fit_transform(independentVariable))
print(independentVariable)
# Incase the categorical variables where many, then we could create an list with the columns
#  names that are categorical and replace [0] with the list object.

# Dependent Variable
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dependentVariable = label.fit_transform(dependentVariable)

# Splitting Into Test and Train Sets
from sklearn.model_selection import train_test_split
independentVariableTrain, independentVariableTest, dependentVariableTrain ,dependentVariableTest = train_test_split(independentVariable,dependentVariable,test_size=0.2,random_state=1)

# Feature Scaling
# Do not Scale dummy variables

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
independentVariableTrain[:,3:] = scaler.fit_transform(independentVariableTrain[:,3:])

independentVariableTest[:,3:] = scaler.transform(independentVariableTest[:,3:])

