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
#https://asset.kompas.com/crops/nXDyHLD7t4Xrnh-pXSIz7eR82S4=/0x39:668x484/750x500/data/photo/2021/08/17/611b57243b494.jpeg
#https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg
#https://asset.kompas.com/crops/S08EJQCN7FCZBX7feGnlOfL3wQg=/192x128:1728x1152/750x500/data/photo/2021/02/04/601c1ff4c67b8.jpg
#https://asset-a.grid.id/crop/0x0:0x0/700x465/photo/2019/12/30/4149872808.jpg
#https://asset.kompas.com/crops/FyXNOOJGN1XMvLVF0fJIhamCIvY=/0x0:750x500/750x500/data/photo/2021/05/13/609d41bc692d8.jpg
#https://cdn0-production-images-kly.akamaized.net/SAwzqKST_4PyGfi-5_koGptDHC8=/640x640/smart/filters:quality(75):strip_icc():format(jpeg)/kly-media-production/product_images/6352/original/087774700_1605879012-dfba11d13e4ccede0b090aebaef683a2.jpg

#Test Foto Tinja
#https://2.bp.blogspot.com/-3N_KfaNWxz4/V5oaqCgQX3I/AAAAAAAACeA/XBTnU9ptuk0Dfx8xwAxrlERncZ1KuPsZgCLcB/s320/cacing.jpg

