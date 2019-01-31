#OS for file path.
import os

#OpenCV2 for image processing.
import cv2

#numpy for numerical and scientific computing.
import numpy as np

#PIL - Python Image Library.
from PIL import Image

#If path exists, select it else create it.
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#Create Local Binary Pattern Histograms for Face Recognizing.
recognizer = cv2.face.LBPHFaceRecognizer_create()

#"haarcascade_frontalface_default.xml" is pre-built frontal face training model from OpenCV using for Face detection.
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

#Set path as dataSet.
path = 'dataSet'

#Function to get the images with label data.
def getImagesAndLabels(path):

    #Get all file path.
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    #For faces.
    faceSamples = []

    #For IDs.
    ids = []

    #Loop for all the file path.
    for imagePath in imagePaths:

        #Convert the image to gray scale.
        PIL_img = Image.open(imagePath).convert('L')

        #Convert PIL image to Numpy Array.
        img_numpy = np.array(PIL_img, 'uint8')

        #Image ID.
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        #Get face from training images.
        faces = detector.detectMultiScale(img_numpy)

        #Loop for faces and add the corresponding IDs.
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceSamples, ids

#Get faces and IDs.
faces,ids = getImagesAndLabels('dataset')

#Training model using faces and IDs.
recognizer.train(faces, np.array(ids))

#Checking for training folder.
assure_path_exists('trainer/')

#Save the model into trainer.yml
recognizer.save('trainer/trainer.yml')