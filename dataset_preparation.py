import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import matplotlib.pyplot as plt
import mtcnn
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

model = load_model('facenet_keras.h5')

# summarize input and output shape

#print(model.inputs)
#print(model.outputs)

# function for face detection with mtcnn

def extract_face(filename, required_size=(160, 160)):
	"""
	extract a single face from a given photograph and 
	acts as the input to facenet model
	"""
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	
	# convert to array
	pixels = asarray(image)
	
	# create the detector, using default weights
	detector = MTCNN()
	
	# detect faces in the image -- does it by passing the images through p-net, r-net and o-net
	results = detector.detect_faces(pixels)
	print(results[0]['confidence'])
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	## Creates an image memory from an object exporting the array interface 
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# specify folder to plot

#folder = 'faces-dataset/train/ben_afflek/'
#i = 1
# enumerate files
#for filename in os.listdir(folder):
	# path
#	path = folder + filename
	# get face
#	face = extract_face(path)
#	print(i, face.shape)
	# plot
#	plt.subplot(2, 7, i)
#	plt.axis('off')
#	plt.imshow(face)
#	i += 1
#plt.show()

def load_faces(directory):
	"""
	load images and extract faces for all images in a directory
	"""
	faces = list()
	# enumerate files
	for filename in os.listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces

def load_dataset(directory):
	"""
	The load_dataset() function takes a directory name 
	such as ‘faces-dataset/train/‘ and detects faces for 
	each subdirectory, assigning labels to each detected face.
	"""
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in os.listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not os.path.isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)


if __name__ == '__main__':
	# load train dataset
	trainX, trainy = load_dataset('faces-dataset/train1/')
	print(trainX.shape, trainy.shape)
	# load test dataset
	#testX, testy = load_dataset('faces-dataset/val/')
	#print(testX.shape, testy.shape)
	# save arrays to one file in compressed format
	np.savez_compressed('faces-dataset.npz', trainX, trainy)
	#np.savez_compressed('faces-dataset.npz', trainX, trainy, testX, testy)