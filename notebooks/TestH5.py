import os
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = './raw_data/9426434L.png'
path_model = './modelo_franco_MobileNet121.h5'

def prepro(path):
    # Prepro
    imagen = Image.open(path)
    imagen = imagen.convert('RGB')
    imagen_np = (np.array(imagen)) / 255
    imagen_exp = np.expand_dims(imagen_np, 0)
    return (imagen_exp)

def load_model(path_model):
    modelito = tf.keras.models.load_model(path_model, compile=False)
    return modelito

model = load_model(path_model)
imagen_exp2 = prepro(path)
resultado = model.predict(imagen_exp2)
print(resultado)
