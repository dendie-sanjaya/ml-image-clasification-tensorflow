#start import lib
import os; 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';  
import requests 
import numpy as np
import tensorflow as tf
import pathlib
from io import BytesIO 
from PIL import Image
from tensorflow.keras.models import load_model
import json;
from flask import abort, Flask, jsonify, redirect, request, url_for
from cfg import *


#MODEL_PATH = 'model/GCH/clasification_flower.h5'
MODEL_PATH = CONFIG_MODEL_PATH
model = load_model(MODEL_PATH,compile=False)

app = Flask(__name__)

@app.errorhandler(400)
def bad_request(e):
    return response_api({
        'code': 400,
        'message': 'Ada kekeliruan input saat melakukan request.',
        'data': None
    })

@app.errorhandler(500)
def internal_server_error(e):
    return response_api({
        'code': 500,
        'message': 'Mohon maaf, ada gangguan pada server kami.',
        'data': None
    })

def response_api(data):
    return (jsonify(data));

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

  #format(class_names[np.argmax(score)], 100 * np.max(score))
  return format(class_names[np.argmax(score)]),(100 * np.max(score))

@app.route('/')
def root():
    return 'RESTful API Mechine Learning Image Clasification'

@app.route('/analysis', methods=['GET'])
def analysis():
    url_foto = request.args.get('url_foto');
    key = request.args.get('key');

    if((len(url_foto) > 7) and (key == 'pwd123')):
      pr = predictions(url_foto);
      probabilitas = pr[1]
      prediksi = pr[0] 
      respon = {
          'code': 200,
          'message': 'Foto berhasil di analisa',
          'probabilitas': probabilitas,
          'prediksi' : prediksi,
          'url_foto': url_foto,
          'key': key
      }
    else:
      respon = {
          'code': 401,
          'message': 'API Key Tidak Valid atau Foto Tidak Ditemukan',
          'probabilitas': 0,
          'prediksi' : 0,
          'url_foto': url_foto,
          'key': key
      }


    return response_api(respon)

app.run(debug=True,threaded=True)    