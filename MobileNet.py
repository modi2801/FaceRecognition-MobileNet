#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import MobileNet
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.models import Model
import cv2
import numpy as np
from PIL import Image


# In[2]:


img_rows = 224
img_cols = 224  
model = MobileNet(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))


# In[3]:


for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[4]:


model.layers[0].trainable


# In[5]:


model.output


# In[6]:


for layer in model.layers:
    layer.trainable = False


# In[7]:


model_output = model.output


# In[8]:


model_output = Flatten()(model_output)


# In[9]:


model_output = Dense(units = 512 , activation='relu' )(model_output)
model_output = Dense(units= 256, activation='relu' )(model_output)
model_output = Dense(units= 1 , activation='sigmoid' )(model_output)


# In[10]:


model= Model(inputs=model.input , outputs = model_output)


# In[11]:


model.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])


# In[12]:


model.summary()


# In[13]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'train/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'validation/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=70,
        epochs=1,
        validation_data=test_set,
        validation_steps=10)


# In[14]:


model.save('face_recog_mobilenet.h5')


# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:


from keras.models import load_model
from keras.preprocessing import image
model = load_model('face_recog_mobilenet.h5')

face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

labels_dict={0:'yash' ,1:'mom'}
color_dict={0:(0,255,0),1:(0,0,255)}


# In[38]:


while (True):
    
    ret,img=cap.read()
    faces=face_clsfr.detectMultiScale(img)
    
    for (x,y,w,h) in faces:
        face_img = img[y:y+w , x:x+w]
        reshaped = cv2.resize(face_img , (224,224))
        reshaped = image.img_to_array(reshaped)
        reshaped = np.expand_dims(reshaped, axis=0)
        result=model.predict(reshaped)
        
        label=np.argmax(result,axis=1)[0]
        
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img,labels_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break

cv2.destroyAllWindows()


# In[37]:


cap.release()


# In[ ]:




