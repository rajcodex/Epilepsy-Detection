import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Flatten, Reshape, Dropout

class ProposedModelFrameWork:
    
    def __init__(self, classes,input_shape):
        self.classes=classes
        self.input_shape=input_shape
        self.model= Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        self.model.add(Conv1D(64, kernel_size=3, activation='relu'))
        self.model.add(Conv1D(64, kernel_size=3, activation='relu'))
        self.model.add(Conv1D(64, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        # Flatten layer to prepare for GRU
        self.model.add(Flatten())
        # Reshape to fit GRU input shape
        self.model.add(Reshape((-1, 64)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))

        # First GRU layer
        self.model.add(GRU(units=64, return_sequences=True))

        # Second GRU layer
        self.model.add(GRU(units=128, return_sequences=False))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))

        # Dense layers for classification
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
        self.model.add(Dense(self.classes, activation='softmax'))  
    

    def train(self,X_train, y_train , epoch , batch , validation):

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history=self.model.fit(X_train, y_train, epochs=epoch, batch_size= batch , validation_split= validation)

        return history
    def predict(self,X_test):
        return self.model.predict(X_test) 
    
    def evaluate(self,X_test,y_test):
        return self.model.evaluate(X_test,y_test)

    
    
 

