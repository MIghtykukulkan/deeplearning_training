# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nsepy import get_history
from datetime import date

stockcode = "TNPETRO"
data = get_history(symbol=stockcode, start=date(2016,1,1), end=date.today())

# Importing the training set
#dataset_train = pd.read_csv('hexaware_train.csv')
dataset_train = data
training_set = dataset_train.iloc[:, 2:3,].values



# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 820):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#regressor.save('tatamotors.h5')



# Part 3 - Making the predictions and visualising the results

#doenload -link - 
# Getting the real stock price of 2017
#dataset_test = pd.read_csv('tatamotors.csv')
dataset_test = data
real_stock_price = dataset_test.iloc[:, 2:3].values

# Getting the predicted stock price of 2017

#dataset_total = pd.concat((dataset_train['Open Price'], dataset_test['Open Price']), axis = 0)
inputOriginal = real_stock_price[ -60:]
inputs = sc.transform(inputOriginal)

inputs = np.reshape(inputs, (1, 60, 1))
predicted_stock_price = regressor.predict(inputs)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(inputOriginal[59], predicted_stock_price[0])

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Hexaware Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Hexaware Stock Price')
plt.title('Hexaware Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Hexaware Stock Price')
plt.legend()
plt.show()
