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

sc = StandardScaler()

dataset = pd.read_csv('digit-recognizer/train.csv')
x_train = dataset.iloc[:, 1:785].values
x_train = sc.fit_transform(x_train)

numberEncoder = LabelEncoder()
y_train = dataset.iloc[:, 0].values
y_train = numberEncoder.fit_transform(y_train)
#y_train = sc.fit_transform(y_train)

testset = pd.read_csv('digit-recognizer/test.csv')
x_test = testset.iloc[0: 28000].values

#y_test = sc.transform(y_test)


        
# Initialising the ANN
classifier = Sequential()

classifier.add(Dropout(0.2))

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784))


# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 16, epochs = 100)

classifier.save('mnist.h5')


classifier = load_model('mnist.h5')
y_pred = classifier.predict(x_test)
y_pred = classifier.predict(np.array([x_train[3]]))
print(y_pred.tolist())

pixels = x_train[3127].reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
print(y_pred[3127])
        
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.9)
predictions = []
for i in range(len(y_pred)):
    print(y_pred[i], i)
    predictions.append([i+1,y_pred[i].tolist().index(True)])

y_pred = np.array(predictions)
pd.DataFrame(y_pred).to_csv("foo.csv", header=["ImageId","Label"], index=False)