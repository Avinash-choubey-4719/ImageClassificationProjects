import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

img_size=(128,128)

model = keras.models.load_model('model1_catsVSdogs_10epoch.h5')
model.summary()

results={
    0:'cat',
    1:'dog'
}
from PIL import Image
import numpy as np
im=Image.open("/home/avinashtechhack/Downloads/test1/2.jpg")
im=im.resize(img_size)
im=np.expand_dims(im,axis=0)
im=np.array(im)
im=im/255
pred=np.argmax(model.predict(im), axis = 1)
print(pred)
