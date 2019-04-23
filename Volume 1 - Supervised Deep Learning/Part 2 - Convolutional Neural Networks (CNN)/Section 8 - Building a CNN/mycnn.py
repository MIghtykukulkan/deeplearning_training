# -*- coding: utf-8 -*-

#importing libraries

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#builing convolution layers

classifier = Sequential()

#convolution layer
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation = 'relu'))
#max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#flattern
classifier.add(Flatten())

#adding fully conected layer i.e., ANN
classifier.add(Dense(units=128, activation = 'relu'))

classifier.add(Dense(units=1, activation = 'sigmoid'))

#compilation

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#data preprocessing keras
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), #shape defined in conv2d
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)

classifier.save('my_model.h5')


import numpy as np
from keras.preprocessing import image
testimage = image.load_img('mydog.jpg', target_size = (64,64))
testimage = image.img_to_array(testimage)
testimage = np.expand_dims(testimage, axis=0)

result = classifier.predict(testimage)
training_set.class_indices

from keras.models import load_model
model = load_model('my_model.h5')

from keras.utils import plot_model
plot_model(model, to_file='model.png')

           

