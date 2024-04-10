# Importing the needed Libraries
import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

#  Importing the data
Dataset = pd.read_csv('Churn_Modelling.csv')
Features = Dataset.iloc[:,3:-1].values
Label = Dataset.iloc[:,-1].values

# Encoding the Categorical Columns
# Gender
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Features[:,2] = label.fit_transform(Features[:,2])

# Geography
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
objectClass = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')
Features = np.array(objectClass.fit_transform(Features))

# Splitting Into Test and Train Sets
from sklearn.model_selection import train_test_split
indTrain, indTest, depTrain ,depTest = train_test_split(Features,Label,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
indTrain = scaler.fit_transform(indTrain)
indTest = scaler.transform(indTest)

# Building the Artificial Neural Network
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding an input layer and a hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Second Hidden Layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Output Layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN with the adam optimizer and binary_crossentropy loss
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics =['accuracy'])

# Train the ANN
ann.fit(indTrain,depTrain,batch_size =32,epochs =100)

# Predicting a Single Observation
predProb = ann.predict(scaler.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))
print(predProb > 0.5)

# Predicting the test set 
pred = ann.predict(indTest)
pred = (pred > 0.5)
print(np.concatenate((depTest.reshape(len(depTest),1),pred.reshape(len(depTest),1)),1))

# Creating the Confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
matrix = confusion_matrix(depTest,pred)
accuracy = accuracy_score(depTest,pred)
print(matrix)
print(accuracy)









