# Import tensorflow, matplotlib, numpy and the necessary layers
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

# Import the necessary functions
from deep_learning_model import functional_model, MyCustomModel
from my_utils import display_some_examples

# Define the model in a sequential way

seq_model=tensorflow.keras.Sequential([
    # Our data is 28 by 28 and in gray scale. Hence the shape (28,28,1)
    Input(shape=(28,28,1)),
    
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D(),
    BatchNormalization(),

    Conv2D(128,(3,3), activation='relu'),
    MaxPool2D(),
    BatchNormalization(),

    GlobalAvgPool2D(),
    Dense(64, activation='relu'),

    # There is 10 classes in the MNIST data set hence the 10 hidden units
    Dense(10, activation='softmax')
])

if __name__=='__main__':
    # Import the dataset
    (x_train, y_train), (x_test, y_test)=tensorflow.keras.datasets.mnist.load_data()

    # Check the dimensions of the dataset
    print("x_train.shape= ", x_train.shape)
    print("y_train.shape= ", y_train.shape)
    print("x_test.shape= ", x_test.shape)
    print("y_test.shape= ", y_test.shape)

    if False:
        display_some_examples(x_train, y_train)
    
    # Normalize the dataset, 255 being the max value
    # First, convert the data to float otherwise it will remain integer and that is not what we want

    x_train=x_train.astype('float32')/255
    x_test=x_test.astype('float32')/255

    # Expand dimensions of the dataset to match the input layer of the model
    x_train=np.expand_dims(x_train,axis=-1)
    x_test=np.expand_dims(x_test, axis=-1)

    # Check the new dimensions of the dataset
    print("x_train.shape= ", x_train.shape)
    print("y_train.shape= ", y_train.shape)
    print("x_test.shape= ", x_test.shape)
    print("y_test.shape= ", y_test.shape)

    # Sparse does not need one hot encoding while categorical does.

    # Model compilation
    seq_model.compile(loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics='accuracy')

    # Model training
    seq_model.fit(x_train, y_train, batch_size=64, epochs=3,validation_split=0.2)

    # Model evaluation
    seq_model.evaluate(x_test, y_test, batch_size=64)

    