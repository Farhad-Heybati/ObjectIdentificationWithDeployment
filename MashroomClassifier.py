# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 07:28:47 2020

@author: chbhakat
"""
# pip install keras
# pip install tensorflow

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
# from tensorflow.keras.applications.resnet_v2 import ResNet152V2
# from tensorflow.keras.applications.resnet_v2 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pickle

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'C:/Users/chbhakat/source/datascience/images/classes/Train'
valid_path = 'C:/Users/chbhakat/source/datascience/images/classes/Test'

# add preprocessing layer to the front of VGG
# vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
vgg = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# vgg = ResNet152V2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  

  
  # useful for getting number of classes
folders = glob('C:/Users/chbhakat/source/datascience/images/Train/*')
  

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/chbhakat/source/datascience/images/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('C:/Users/chbhakat/source/datascience/images/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

# model.save('Mashroom_Classifier_VGG16.h5')
model.save('Mashroom_Classifier_InceptionV3_20Epochs.h5')
# model.save('Mashroom_Classifier_ResNet152v2.h5')

#-----------------------------------------------------------------------
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('Mashroom_Classifier_ResNet152v2.h5')

# img=image.load_img('C:/Users/chbhakat/source/datascience/images/classes/Validation/Morchella/40254.jpg',target_size=(224,224))
# img=image.load_img('C:/Users/chbhakat/source/datascience/images/classes/Validation/Pleurotus ostreatus/32744.jpg',target_size=(224,224))
# img=image.load_img('C:/Users/chbhakat/source/datascience/images/classes/Validation/Boletus edulis/59471.jpg',target_size=(224,224))
img=image.load_img('C:/Users/chbhakat/source/datascience/images/classes/Validation/Hypomyces lactifluorum/54026.jpg',target_size=(224,224))
x=image.img_to_array(img)
print(x)

x.shape

x=x/255
print(x)

import numpy as np
x=np.expand_dims(x,axis=0)
print(x)
print(x.shape)
img_data=preprocess_input(x)
print(img_data)
img_data.shape



model.predict(img_data)

a=np.argmax(model.predict(img_data), axis=1)
print(a)

# -----------------------------------------------------------------------
# f = request.files['file']

# Save the file to ./uploads
basepath = os.path.dirname(__file__)
file_path = os.path.join(
    basepath, 'uploads', secure_filename(f.filename))
f.save(file_path)       
