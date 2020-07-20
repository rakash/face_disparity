import numpy as np 
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	#print(f'face_pixels_size: {face_pixels.size}')
	#print(f'face_pixels_ndim: {face_pixels.ndim}')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = tf.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


if __name__ == '__main__':
	# load the face dataset
	data = np.load('faces-dataset.npz')
	#trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	trainX, trainy = data['arr_0'], data['arr_1']
	print('Loaded: ', trainX.shape, trainy.shape)#, testX.shape, testy.shape)
	## load the facenet model
	model = load_model('model.h5')

	## Vgg model
	#model = load_model('vgg_face_weights.h5')

	print('Loaded Model')

	# convert each face in the train set to an embedding
	newTrainX = list()
	for face_pixels in trainX:
		#print(f'fppx: {face_pixels.shape}')
		#print(f'fpptx: {trainX.shape}')
		embedding = get_embedding(model, face_pixels)
		newTrainX.append(embedding)
	newTrainX = np.asarray(newTrainX)
	print(newTrainX.shape)

	# convert each face in the test set to an embedding
	#newTestX = list()
	#for face_pixels in testX:
#		embedding = get_embedding(model, face_pixels)
	#	newTestX.append(embedding)
	#newTestX = np.asarray(newTestX)
	#print(newTestX.shape)

	# save arrays to one file in compressed format	
	#np.savez_compressed('faces-embeddings.npz', newTrainX, trainy, newTestX, testy)
	np.savez_compressed('faces-embeddings_kamal.npz', newTrainX, trainy)
