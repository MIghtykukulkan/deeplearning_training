# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()

dataset = pd.read_csv('digit-recognizer/train.csv')
x_train = dataset.iloc[:, 1:785].values
y_train = dataset.iloc[:, 0].values
#y_train = sc.fit_transform(y_train)

testset = pd.read_csv('digit-recognizer/test.csv')
x_test = testset.iloc[:784].values

#y_test = sc.transform(y_test)


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(np.array([x_test[3]]))
print(y_pred)

