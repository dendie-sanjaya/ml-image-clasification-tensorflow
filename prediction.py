#start import libray
import os; 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';  
import requests 
import numpy as np
import tensorflow as tf
import pathlib
import PIL
from io import BytesIO 
from cfg import *

#from PIL import Image


from tensorflow.keras.models import load_model

#step 1 start input model 
print("\n1. Load Model  \n");
MODEL_PATH = CONFIG_MODEL_PATH
model = load_model(MODEL_PATH,compile=False)
# end step 1 


#step 2 pembuatan prediksi
print("2. Prediksi \n");

def predictions(source_test):
  img_height = 150
  img_width = 150
  class_names = CONFIG_CLASS_NAME

  foto_test =  source_test
  foto_test_path = tf.keras.utils.get_file(CONFIG_FOTO_NAME_TEST, origin=foto_test)
  print("_____\nFoto Test Path :",foto_test_path,"\n")

  img = tf.keras.utils.load_img(
      foto_test_path, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) 

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  tf.io.gfile.remove(CONFIG_PATH_NAME_TEST)

  print("_____\nFoto Test yang di prediksi :",foto_test,"\n")
  print("_____ Hasil Prediksi Gambar ini seperti {} dengan probabiltas {:.2f} percent cocok.".format(class_names[np.argmax(score)], 100 * np.max(score)))


def source_test():
  source_test_url = input("Silakan Inputkan Gambar yang akan di prediksi ! ")
  if(source_test_url != ""):
    predictions(source_test_url)
  else:
    source_test()

source_test()

#Test Foto Bunga
#https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg
#https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg
