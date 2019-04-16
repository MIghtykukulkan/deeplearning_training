# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

features = dataset.iloc[:, 3:13].values
label = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_country = LabelEncoder()
features[:, 1] = labelEncoder_country.fit_transform(features[:, 1])
labelEncoder_gender = LabelEncoder()
features[:, 2] = labelEncoder_gender.fit_transform(features[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
features = onehotencoder.fit_transform(features).toarray()
features  = features[:, 1:] #removing one coloumn to avoid dummy variable trap 



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
X_train =standardScaler.fit_transform(X_train)
X_test  = standardScaler.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier  = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#predicting a single function
y_pred_single = classifier.predict(standardScaler.transform(np.array([[0.0,0,600, 1, 40, 3, 60000, 2,  1, 1, 50000]])))

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV
def build_classifier():    
    classifier  = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier 

classifier = KerasClassifier(build_fn = build_classifier,  batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1) 
mean = accuracies.mean()
variance = accuracies.std()

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):    
    classifier  = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier 
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [25, 32],
              'epochs':[100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring='accuracy',
                           cv=10) 
grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
#parameter tuning

