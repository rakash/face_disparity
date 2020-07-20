import numpy as np
import os
from cv2 import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import model_from_json
from os import listdir
from tensorflow.keras.models import load_model
import tensorflow as tf 
from dataset_preparation import extract_face
from face_embeddings import get_embedding
from face_Alignment import face_align
import time 
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def normEuclideanDistance(source_representation, test_representation):
    in_encoder = Normalizer(norm='l2')
    source_representation = in_encoder.transform(source_representation.reshape(1, -1))
    test_representation = in_encoder.transform(test_representation.reshape(1, -1))
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def findCosineSimilarity(source_representation, test_representation):
    return 1 - (cosine(source_representation, test_representation))

def normCosineSimilarity(source_representation, test_representation):
    in_encoder = Normalizer(norm='l1')
    source_representation = in_encoder.transform(source_representation.reshape(1, -1))
    test_representation = in_encoder.transform(test_representation.reshape(1, -1))
    return 1 - (cosine(source_representation, test_representation))

def preprocess_image(image_path):
    ## Loads an image into PIL format.
    img = load_img(image_path, target_size=(160, 160))
    ## Converts a PIL Image instance to a Numpy array and adds channels.
    img = img_to_array(img)
    ## subtracts the mean RGB channels of the imagenet dataset. -- normalizes
    img = preprocess_input(img)
    ## expand the dimensions for each image
    img = np.expand_dims(img, axis=0)
    return img

def embedding():
    """
    get embeddings based on process_image
    """
    pictures = "faces-dataset/train1/kamal/"
    names = dict()
    model = load_model('facenet_keras.h5')
    for file in os.listdir(pictures):
        name, extension = file.split(".")
        img = preprocess_image('faces-dataset/train1/kamal/%s.jpg' % (name))
        representation = model.predict(img)[0,:]
        names[name] = representation
	
    print("embeddings retrieved successfully")
    print(f'cosine distance : {cosine(names["09"], names["10"])}')
    print(f'Euclidean distance : {findEuclideanDistance(names["09"], names["10"])}')
    print(f'cosine similarity : {1 - (cosine(names["09"], names["10"]))}')
    sims = 1 - (cosine(names["09"], names["10"]))
    print(f'disparity: {1 - sims}')
    return names

def embedding2(alignd='faces-dataset/train1/kamal/', embd='faces-dataset/train1/kamal_aligned/'):
    """
    get embeddings based on mtccn extraction and keras pretrained
    """
    face_align(alignd, embd)
    model = load_model('facenet_keras.h5')
    names = dict()
    for filename in os.listdir(embd):
        name, extension = filename.split(".")
        print(f'name: {name}')    
        testX = extract_face(embd+'/'+filename)
        #np.savez_compressed('test_dataset.npz', testX)
        embedding = get_embedding(model, testX)
        embedding = np.asarray(embedding)
        names[name] = embedding
    img1= "09_aligned"
    img2= "10_aligned"
    print("embeddings retrieved successfully")
    print(f'cosine distance : {cosine(names[img1], names[img2])}')
    print(f'Euclidean distance : {findEuclideanDistance(names[img1], names[img2])}')
    print(f'Normalised Euclidean distance : {normEuclideanDistance(names[img1].reshape(-1, 1), names[img2].reshape(-1, 1))}')
    print(f'cosine similarity : {findCosineSimilarity(names[img1], names[img2])}')
    print(f'Normalised cosine similarity : {normCosineSimilarity(names[img1], names[img2])}')
    sims = findCosineSimilarity(names[img1], names[img2])
    print(f'Cosine disparity: {1 - sims}')
    return names

def plot():
    """
    function to plot the extracted faces and to check for alignment
    """
    folder = 'faces-dataset/train1/kamal1/'
    i = 1
    for filename in os.listdir(folder):
	    # path
        path = folder + filename
	    # get face
        face = extract_face(path)
        print(i, face.shape)
        print(filename, face.shape)
	    # plot
        plt.subplot(2, 7, i)
        plt.axis('off')
        plt.imshow(face)
        i += 1
        plt.show()

if __name__ == '__main__':
    start_time = time.time()
    embedding2()
    #plot()
    print(f'Recognition took {time.time() - start_time} seconds to run')