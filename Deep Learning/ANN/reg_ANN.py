import numpy as np
import pandas as pd 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

tf.__version__

# Importing Dataset and Creating matrixs of Features
dataset = pd.read_csv('Folds.csv')
indVariable = dataset.iloc[:,:-1].values
depVariable = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
indTrain,indTest,depTrain,depTest = train_test_split(indVariable,depVariable,test_size=0.2, random_state=0)

# Building the ANN
#Intialiazing the ANN
ann = tf.keras.models.Sequential()

#  Adding the hidden layers
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1))

# Training the ANN
ann.compile(optimizer ='adam',loss = 'mean_squared_error')
ann.fit(indTrain,depTrain, batch_size = 32,epochs =100)

# Predicting using the ANN
depPred = ann.predict(indTest)
np.set_printoptions(precision=2)
print(np.concatenate((depPred.reshape(len(depPred),1),depTest.reshape(len(depTest),1)),1))

# Checking for Accuracy
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error
mse = mean_squared_error(depTest,depPred) 
mape = mean_absolute_percentage_error(depTest,depPred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Percentage Error: {mape}')
