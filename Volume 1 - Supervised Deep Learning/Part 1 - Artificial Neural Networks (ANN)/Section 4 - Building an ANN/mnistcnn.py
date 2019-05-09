# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
#from keras.datasets import mnist

#mn = mnist.load_data()

sc = StandardScaler()

dataset = pd.read_csv('digit-recognizer/train.csv')
x_train = dataset.iloc[:, 1:785].values
x_train = sc.fit_transform(x_train)
x_train = x_train.reshape(-1 , 28 , 28 , 1)

numberEncoder = LabelEncoder()
y_train = dataset.iloc[:, 0].values
y_train = numberEncoder.fit_transform(y_train)
#y_train = sc.fit_transform(y_train)

xTrain, xTest, yTrain, yTest = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

testset = pd.read_csv('digit-recognizer/test.csv')
x_test = testset.iloc[0: 28000].values
x_test = x_test.reshape(-1 , 28 , 28 , 1)
#y_test = sc.transform(y_test)

        
# Initialising the ANN
classifier = Sequential()

#convolution layer
classifier.add(Convolution2D(64, (3, 3), input_shape=(28,28, 1), activation = 'relu'))
#max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
#convolution layer - 2 note that the input shape is not requied to add second conv layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
#max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

#flattern
classifier.add(Flatten())

# Adding the second hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])

# Fitting the ANN to the Training set
classifier.fit(xTrain, yTrain,
          batch_size=16,
          epochs=50,
          verbose=1,
          validation_data=(xTest, yTest))


classifier.save('mnistcnn.h5')


classifier = load_model('mnistcnn.h5')
y_pred = classifier.predict(x_test)
y_pred = classifier.predict(np.array([x_train[3]]))
print(y_pred.tolist())

pixels = x_train[1151]
plt.imshow(pixels, cmap='gray')
plt.show()
print(y_pred[1551])
tmp = y_pred[9652]
        
y_pred = classifier.predict(x_test)
y_pred_bool = (y_pred > 0.5)
predictions = []
for i in range(len(y_pred_bool)):
    print(y_pred_bool[i], i)
    predictions.append([i+1,y_pred_bool[i].tolist().index(True)])

y_pred = np.array(predictions)
pd.DataFrame(y_pred).to_csv("foo.csv", header=["ImageId","Label"], index=False)