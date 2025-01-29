# Importing Libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np


#Data Preprocessing
# Train Set
trainGen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

trainSet = trainGen.flow_from_directory(
    'dataset/training_set',
    target_size = (100,100),
    batch_size = 32,
    class_mode = 'binary'
)

# Test Set
testGen = ImageDataGenerator(
    rescale = 1./255
)


testSet = testGen.flow_from_directory(
    'dataset/test_set',
    target_size = (100,100),
    batch_size = 32,
    class_mode = 'binary'
)

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Convolution
cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size = 3, activation = 'relu', input_shape = [100,100,3]))

# Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

# Second Convolution Layer with Max Pooling
cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

#  Full Connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

# Output Layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

# Training the CNN
cnn.fit(x=trainSet, validation_data=testSet, epochs = 25)

# Making A Single Observation

import keras.src.utils as image
testImage = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', 
                           target_size= (100,100))
testImage = image.img_to_array(testImage)
testImage = np.expand_dims(testImage, axis= 0 )
result = cnn.predict(testImage)
print(testSet.class_indices)

if result[0][0] == 1:
    prediction = 'Dog'
else:
    Prediction = 'Cat'

print(prediction)