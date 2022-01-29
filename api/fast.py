from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
#import pytz
#import joblib
import numpy as np
#from tensorflow.keras import models
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = 'raw_data/9426434L.png'
path_model = 'api/modelo_franco_MobileNet121.h5'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict():
    # Prepro
    imagen = Image.open(path)
    imagen = imagen.convert('RGB')
    imagen_np = (np.array(imagen)) / 255
    imagen_exp = np.expand_dims(imagen_np, 0)
    model = tf.keras.models.load_model(path_model, compile=False)
    prediccion = model.predict(imagen_exp)
    aaaa = f"{np.argmax(prediccion, axis=1)}"
    return {"prediction": aaaa}
