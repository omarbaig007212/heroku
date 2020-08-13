from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import spacy
import pickle
import random
import sys,fitz




# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'nlp_model'

# Load your trained model
nlp_model=spacy.load('nlp_model')
model = nlp_model
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    fname = img_path
    doc = fitz.open(fname)
    text = ""
    preds = ""
    search = ""
    for page in doc:
        text = text + str(page.getText())
    tx = " ".join(text.split('\n'))
    print(tx)
    doc = nlp_model(tx)
    for ent in doc.ents:
        print(f'{ent.label_.upper():{30}}- {ent.text}')
        preds = preds + "\n" + f'{ent.label_.upper():{30}}- {ent.text}'
        search = search + " " + f'{ent.label_.upper():{10}}- {ent.text}' + " "
    with open('buttonpython\\search.txt', 'a') as q:
        q.write("\n")
        q.write(search)
        q.write("\n")
        q.close()
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
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
        with open('buttonpython\\test.txt','a') as f:
            f.write("\n")
            f.write(preds)
            f.write("\n")
            f.close()
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

