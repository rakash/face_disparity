# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
from cv2 import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import os
from PIL import Image
import math
from numpy import asarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shapepredictor", required=True,
#help="path to facial landmark predictor")
#ap.add_argument("-i", "--image", required=True,
#help="path to input image")
#args = vars(ap.parse_args())

def face_align(directory, embd):

    """
    Aligns the face
    """
    # initialize dlib's face detector (HOG+linear classifier) and then create 
    # the facial landmark predictor and the face aligner
    
    detector = dlib.get_frontal_face_detector()
    ### this is a pretrained model- trained on a labeled set of facial landmarks on images. This maps the location of (x,y) coordinates
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    directory = directory
    names = dict()
    for filename in os.listdir(directory):
        name, extension = filename.split(".")
        print(f'name: {name}')

        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(directory+'/'+name+'.jpg')
        image = imutils.resize(image, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # show the original input image and detect faces in the grayscale
        # image
        cv2.imshow("Input", image)
        rects = detector(gray, 2)
        # loop over the face detections
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            #faceOrig = imutils.resize(image[y:y + h, x:x + w], width=160)
            faceAligned = fa.align(image, gray, rect)
            faceAligned1 = imutils.resize(faceAligned, width=160)
            path = embd+f'{name}_aligned.jpg'
            cv2.imwrite(path, faceAligned1)
            #face1 = Image.fromarray(faceAligned1)
            #face1.save(path)


def face_align_img(img):
    """
    same as above but for each image to pass to MTCNN detector
    """
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    # loop over the face detections
    for rect in rects:
        #(x, y, w, h) = rect_to_bb(rect)
        faceAligned = fa.align(image, gray, rect)
        faceAligned1 = imutils.resize(faceAligned, width=160)
        #fa1 = cv2.imwrite('faceAligned1.jpg', faceAligned1)
        fa1 = Image.fromarray(faceAligned1)
    return fa1


def face_align_mtt():
    detector = MTCNN()
    predictor = dlib.shape_predictor(args["shapepredictor"])
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#convert to array
    pixels = asarray(image)
    # show the original input image and detect faces in the grayscale
    # image
    cv2.imshow("Input", image)
    rects = detector.detect_faces(pixels)
    print(f'rects_: {rects}')
    # extract the bounding box from the first face
    x1, y1, width, height = rects[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
	# extract the face
    face = pixels[y1:y2, x1:x2]
    print(f'face_: {face}')
	# resize pixels to the model size
	## Creates an image memory from an object exporting the array interface
    image = Image.fromarray(face)
    required_size=(160, 160)
    image = image.resize(required_size)
    #image = imutils.resize(image, width=800)
    face_array = asarray(image)

    # loop over the face detections
    #for rect in rects:
     #   print(type(rect))
     #   print(rect)
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
    (x, y, w, h) = rects[0]['box']

    #faceOrig = imutils.resize(image[y:y + h, x:x + w], width=160)
    faceAligned = fa.align(image, gray, face_array)
    # display the output images
    #cv2.imshow("Original", faceOrig)
    cv2.imshow("Aligned", faceAligned)
    cv2.waitKey(0)
    cv2.imwrite('al.jpg', faceAligned) 
    faceAligned1 = imutils.resize(faceAligned, width=160)
    cv2.imwrite('al2.jpg', faceAligned1)

def euclidean_distance(a, b):
	x1 = a[0]; y1 = a[1]
	x2 = b[0]; y2 = b[1]
	
	return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detectFace(img):
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]
    path = folders[0]
    for folder in folders[1:]:
	    path = path + "/" + folder

    face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
    if os.path.isfile(face_detector_path) != True:
	    raise ValueError("Confirm that opencv is installed on your environment! Expected path ",detector_path," violated.")

    face_detector = cv2.CascadeClassifier(face_detector_path)
    faces = face_detector.detectMultiScale(img, 1.3, 5)
	#print("found faces: ", len(faces))

    if len(faces) > 0:
	    face = faces[0]
	    face_x, face_y, face_w, face_h = face
	    img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
	    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    return img, img_gray
    else:
	    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    return img, img_gray
        
def face_align_cv(img_path):
           
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]
    path = folders[0]
    for folder in folders[1:]:
	    path = path + "/" + folder

    face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
    eye_detector_path = path+"/data/haarcascade_eye.xml"
    nose_detector_path = path+"/data/haarcascade_mcs_nose.xml"

    if os.path.isfile(face_detector_path) != True:
	    raise ValueError("Confirm that opencv is installed on your environment! Expected path ",detector_path," violated.")

    face_detector = cv2.CascadeClassifier(face_detector_path)
    eye_detector = cv2.CascadeClassifier(eye_detector_path) 
    nose_detector = cv2.CascadeClassifier(nose_detector_path) 
    
    img = cv2.imread(img_path)
    plt.imshow(img[:, :, ::-1])
    plt.show()
    img_raw = img.copy()
    img, gray_img = detectFace(img)
    eyes = eye_detector.detectMultiScale(gray_img)
    #print("found eyes: ",len(eyes))
    if len(eyes) >= 2:
		#find the largest 2 eye
        base_eyes = eyes[:, 2]
		#print(base_eyes)
        items = []
        for i in range(0, len(base_eyes)):
            item = (base_eyes[i], i)
            items.append(item)
        df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
        eyes = eyes[df.idx.values[0:2]]
		
		#--------------------
		#decide left and right eye
        eye_1 = eyes[0]; eye_2 = eyes[1]
		
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
		
		#--------------------
		#center of eyes
		
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
		#center_of_eyes = (int((left_eye_x+right_eye_x)/2), int((left_eye_y+right_eye_y)/2))
        cv2.circle(img, left_eye_center, 2, (255, 0, 0) , 2)
        cv2.circle(img, right_eye_center, 2, (255, 0, 0) , 2)
		#cv2.circle(img, center_of_eyes, 2, (255, 0, 0) , 2)
		
		#----------------------
		#find rotation direction
		
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 #rotate same direction to clock
            print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 #rotate inverse direction of clock
            print("rotate to inverse clock direction")
		
        cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)
        cv2.line(img,right_eye_center, left_eye_center,(67,67,67),1)
        cv2.line(img,left_eye_center, point_3rd,(67,67,67),1)
        cv2.line(img,right_eye_center, point_3rd,(67,67,67),1)
        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, point_3rd)
        c = euclidean_distance(right_eye_center, left_eye_center)
		#print("left eye: ", left_eye_center)
		#print("right eye: ", right_eye_center)
		#print("additional point: ", point_3rd)
		#print("triangle lengths: ",a, b, c)
		
        cos_a = (b*b + c*c - a*a)/(2*b*c)
		#print("cos(a) = ", cos_a)
        angle = np.arccos(cos_a)
		#print("angle: ", angle," in radian")
        angle = (angle * 180) / math.pi
        print("angle: ", angle," in degree")
		
        if direction == -1:
            angle = 90 - angle
		
        print("angle: ", angle," in degree")

		#rotate image
        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle))
	
    return new_img

def test_cv():
    test_set = ['faces-dataset/train/kamal/02.jpg']
    for instance in test_set:
        alignedFace = face_align_cv(instance)
        plt.imshow(alignedFace[:, :, ::-1])
        plt.show()
        cv2.imwrite('al4.jpg', alignedFace)
        #img, gray_img = detectFace(alignedFace)
        #plt.imshow(img[:, :, ::-1])
        #plt.show()