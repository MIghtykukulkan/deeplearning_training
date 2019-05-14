
import numpy as np
from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras_vggface.utils import preprocess_input
    
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import SGD

# taking facemodel fromVGGFace
face_model = VGGFace(model='vgg16', 
                weights='vggface',
                input_shape=(224,224,3)) 

face_model.summary()

for layer in face_model.layers:
    layer.trainable = False

person_count = 2

last_layer = face_model.get_layer('pool5').output

x = Flatten(name='flatten')(last_layer)
x = Dense(1024, activation='relu', name='fc6')(x)
x = Dense(1024, activation='relu', name='fc7')(x)
out = Dense(person_count, activation='softmax', name='fc8')(x)

custom_face = Model(face_model.input, out)

from keras.preprocessing.image import ImageDataGenerator
batch_size = 2
train_path = 'empdataset/training_set'
eval_path = 'empdataset/test_set'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
                        train_path,
                        target_size=(224, 224),
                        batch_size=2,
                        class_mode='sparse',
                        color_mode='rgb')

valid_generator = valid_datagen.flow_from_directory(
    directory=eval_path,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=2,
    class_mode='sparse',
    shuffle=True,
)



#custom_face.compile(loss='sparse_categorical_crossentropy',
#                         optimizer=SGD(lr=1e-4, momentum=0.9),
#                         metrics=['accuracy'])

#history = custom_face.fit_generator(
#        train_generator,
#        validation_data=valid_generator,
#        steps_per_epoch=49/batch_size,
#        validation_steps=valid_generator.n,
#        epochs=50)

#custom_face.evaluate_generator(generator=valid_generator, steps=5)
        
#custom_face.save('vgg_face.h5')



test_img = image.load_img('test2.jpeg', target_size=(224, 224))
img_test = image.img_to_array(test_img)
img_test = np.expand_dims(img_test, axis=0)
img_test = preprocess_input(img_test)
model = load_model('vgg_face.h5')
predictions = model.predict(img_test)
predicted_class=np.argmax(predictions,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class]
print(predictions)