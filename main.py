# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:59:04 2020

@author: chbhakat
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
# import flasgger
# from flasgger import Swagger 

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
# Swagger(app)

# Model saved with Keras model.save()
MODEL_PATH ='Mashroom_Classifier_InceptionV3_20Epochs.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print("Selected image path",img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    print("imput image array:",x)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    print("predicted result:",preds)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Mashroom is Amanita phalloides"
    elif preds==1:
        preds="The Mashroom is Boletus edulis"
    elif preds==2:
        preds="The Mashroom is Cantharellus cibarius"
    elif preds==3:
        preds="The Mashroom is Hypomyces lactifluorum"
    elif preds==4:
        preds="The Mashroom is Morchella"
    else:
        preds="The Mashroom is Pleurotus ostreatus"
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
