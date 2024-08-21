#start import lib
import os; 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';  
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import time
import pathlib
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#print(tf.__version__)
#end import lib

#start step 1 - import data training dan load data extend tgz
print("\n1. Load Data Training & Load Testing Data \n");
dataset_url = "http://localhost/tensorflow/foto_tinja_sakit.tgz"
data_dir = tf.keras.utils.get_file('foto_tinja_sakit', origin=dataset_url, untar=True)
#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
print("______Lokasi Data Training:",data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("______Jumlah Data Training:",image_count,"\n")
#end

#start step 2 - persiapan training data
print("\n2. Persiapan Training Data \n");

#2.1 pengaturan pengambail pixel gambar yang akan di training
batch_size = 32
img_height = 150
img_width = 150

#2.3 menemukan pengelompokan berdasarkan nama folder

print("\n______Subset Training:\n")
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print("\n______Subset Validation:\n")
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print("\n______Pengelempokan Image:",class_names,"\n")

#2.4 tuning memory untuk training 
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#2.5 standarisai data 
normalization_layer = layers.Rescaling(1./255)
#end: step 2 

#start step 3 - pembuatan model training
print("\n3. Pembuatan Model \n");

num_classes = 6
#num_classes = 6
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  #layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
#end step 3


#start step 4 - Melatih Model
print("\n4. Melatih Model \n");

#epochs=10
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
#end step 4


#step 5 save model hasil training
MODEL_BASE_PATH = "model"
PROJECT_NAME = "GCH"
#SAVE_MODEL_NAME = "clasification_flower.h5"
SAVE_MODEL_NAME = "clasification_tinja.h5"
save_model_path = os.path.join(MODEL_BASE_PATH, PROJECT_NAME, SAVE_MODEL_NAME)
if os.path.exists(os.path.join(MODEL_BASE_PATH, PROJECT_NAME)) == False:
    os.makedirs(os.path.join(MODEL_BASE_PATH, PROJECT_NAME))
    
print('\nMenyimpan Model Hasil Training di {}...'.format(save_model_path))
model.save(save_model_path,include_optimizer=False)
#step 5 save model hasil training







