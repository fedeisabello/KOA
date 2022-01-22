from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
path = '/home/francosimurro/code/leandrocino/KOA/raw_data'


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
def predict(input):
    # Prepro

    image_generator = ImageDataGenerator(rescale=1./255,
                                     preprocessing_function = None,
                                     )
    imagen_a_pro = image_generator.flow_from_directory(batch_size=1,
                                                    directory=path,
                                                    shuffle=False,
                                                    target_size=(224, 224),
                                                    class_mode='categorical')

    imagen_a_pro = imagen_a_pro.reshape(1,224,224,3)

    # Prediccion
    model = joblib.load(
        "/home/francosimurro/code/leandrocino/KOA/raw_data/modelo_lindo.joblib",
        mmap_mode=None)



    return {
             "prediction": model.predict(imagen_a_pro)
            }
