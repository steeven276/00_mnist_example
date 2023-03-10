import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D


# The functional way to define the model. This is a model we want to use in a bigger project.
# Indeed, we could use parameters and provides flexibility
def functional_model():
    my_input=Input(shape=(28,28,1))
    
    x=Conv2D(32, (3,3), activation='relu')(my_input)
    x=Conv2D(64, (3,3), activation='relu')(x)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)

    x=Conv2D(128,(3,3), activation='relu')(x)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)

    x=GlobalAvgPool2D()(x)
    x=Dense(64, activation='relu')(x)

    x=Dense(10, activation='softmax')(x)

    model=tensorflow.keras.Model(inputs=my_input, outputs=x)

    return model


# This is a method that I will learn later

class MyCustomModel(tensorflow.keras.Model):
    def __init__(self)->None:
        super().__init__()
        self.conv1=Conv2D(32, (3,3), activation='relu')
        self.conv2=Conv2D(64, (3,3), activation='relu')
        self.maxpool1=MaxPool2D()
        self.batchnorm1=BatchNormalization()

        self.conv3=Conv2D(128,(3,3), activation='relu')
        self.maxpool2=MaxPool2D()
        self.batchnorm2=BatchNormalization()

        self.globalavgpool1=GlobalAvgPool2D()
        self.dense1=Dense(64, activation='relu')

        self.dense2=Dense(10, activation='softmax')
    
    def call(self, my_input):
        x=self.conv1(my_input)
        x=self.conv2(x)
        x=self.maxpool1(x)
        x=self.batchnorm1(x)
        x=self.conv3(x)
        x=self.maxpool2(x)
        x=self.batchnorm2(x)
        x=self.globalavgpool1(x)
        x=self.dense1(x)
        x=self.dense2(x)

        return x