# -*- coding: utf-8 -*-

#importing libraries

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import numpy as np
from keras.preprocessing import image
#data preprocessing keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import plot_model

#builing convolution layers

classifier = Sequential()

#convolution layer
classifier.add(Convolution2D(64, (3, 3), input_shape=(128,128, 3), activation = 'relu'))
#max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#convolution layer - 2 note that the input shape is not requied to add second conv layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
#max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#flattern
classifier.add(Flatten())

#adding fully conected layer i.e., ANN
classifier.add(Dense(units=128, activation = 'relu'))
classifier.add(Dropout(0.6))
 
 #adding fully conected layer i.e., ANN
classifier.add(Dense(units=128, activation = 'relu'))
classifier.add(Dropout(0.3))

classifier.add(Dense(units=1, activation = 'sigmoid'))

#compilation

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'empdataset/training_set',
        target_size=(128, 128), #shape defined in conv2d
        batch_size=2,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'empdataset/test_set',
        target_size=(128, 128),
        batch_size=2,
        class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=11,
                        epochs=50,
                        validation_data=test_set,
                        validation_steps=4)

classifier.save('emp_model.h5')


testimage = image.load_img('guessme2.jpg', target_size = (128,128))
testimage = image.img_to_array(testimage)
testimage = np.expand_dims(testimage, axis=0)

result = classifier.predict(testimage)
training_set.class_indices


model = load_model('emp_model.h5')


plot_model(model,show_shapes=True, to_file='model.png')

           

