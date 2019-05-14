# -*- coding: utf-8 -*-
#Load trained model
from keras.models import load_model
from keras_vggface import utils
import cv2
import numpy as np
from keras.applications.resnet50 import preprocess_input

image_size = 224
device_id = 0 #camera_device id 

model = load_model('vgg_face.h5')

#make labels according to your dataset folder 
labels = {0: 'arjun', 1: 'vicky'} #and so on


cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(device_id)

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    
    gray_img=cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    faces= cascade_classifier.detectMultiScale(gray_img, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(cam_frame,(x,y),(x+w,y+h),(255,255,0),2)
        roi_color = cam_frame [y:y+h, x:x+w]
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        roi_color = cv2.resize(roi_color, (image_size, image_size))
        image = roi_color.astype(np.float32, copy=False)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image) # or version=2
        preds = model.predict(image)
        predicted_class=np.argmax(preds,axis=1)
        
        labels = dict((v,k) for k,v in labels.items())
        person = predicted_class[0]
        print(person)
        print(labels)
        name = []
        if(person == 0):
            name.append('arjun')
        else:
            name.append('vicky')
            

        cv2.putText(cam_frame,str(name), 
                    (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        
    cv2.imshow('video image', cam_frame)
    key = cv2.waitKey(30)
    if key == 27: # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
