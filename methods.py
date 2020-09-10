from keras import utils
import logging
import json
import pickle
from logging.config import dictConfig
import numpy as np
import cv2
from flask import abort
from PIL import Image
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image as img
from keras.preprocessing.image import load_img, img_to_array
def run_model(img):
	try :
	   model = load_model('model3.h5')
       
	except FileNotFoundError as e :        
	   return abort('Unable to find the file: %s.' % str(e), 503)
	pred = model.predict([img])
	prd_cls = np.argmax(pred)
	return prd_cls
def load_image(filename):
    Testing_Images = []
    img_path=filename
    img1 = cv2.imread(img_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img1, (1024,683))
    Testing_Images.append(new_array)


    Testing_Images = np.array(Testing_Images)
    #x=preprocess_input(x)
    return Testing_Images
def classify(data):
	upload = data
	image = load_image(upload)
	#load_image() is to process image :
	print('image ready')
	try:
		prediction = run_model(image)

		return (json.dumps({"prediction": str(prediction)}))
	except FileNotFoundError as e:
		return abort('Unable to locate image: %s.' % str(e), 503)
	except Exception as e:
		return abort('Unable to process image: %s.' % str(e), 500)






