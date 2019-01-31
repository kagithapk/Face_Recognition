#OpenCV2 for image processing.
import cv2

#OS for file path.
import os

#numpy for numerical and scientific computing.
import numpy as np

#If path exists, select it else create it.
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#Create Local Binary Pattern Histograms for Face Recognizing.
recognizer = cv2.face.LBPHFaceRecognizer_create()

#Check for trainer folder existance.
assure_path_exists("trainer/")

#Getting the trained model.
recognizer.read('trainer/trainer.yml')

#"haarcascade_frontalface_default.xml" is pre-built frontal face training model from OpenCV using for Face detection.
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

font = cv2.FONT_HERSHEY_SIMPLEX

#Starting Video Capturing and 0 indicates the default value for webcam.
cam = cv2.VideoCapture(0);


while(True):

    #Capturing Video Frame.
    ret, img = cam.read();

    #Converting the frame to grayScale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect frames of different sizes, list of faces rectangles.
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);

    #Loop for face detected in each frame.
    for(x,y,w,h) in faces :

        #Creates rectangle around face.
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 4)

        #Recognize the face ID.
        Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        #Check for ID.
        if (Id == 1):
            Id = "Praveen {0:.2f}%".format(round(100 - confidence, 2))
        elif(Id == 2):
            Id = "KPK {0:.2f}%".format(round(100 - confidence), 2)
        elif(Id == 3) :
            Id = "JS R {0:.2f}%".format(round(100 - confidence), 2)

        #Naming the recognized face.
        cv2.rectangle(img, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
        cv2.putText(img, str(Id), (x, y - 40), font, 1, (255, 255, 255), 3)

    #Show video frame with bounded attributes.
    cv2.imshow("Face", img);

    #Press 'q' to quit.
    if(cv2.waitKey(1) == ord('q')) :
        break;

#Close Video Camera.
cam.release()

#Close all the windows.
cv2.destroyAllWindows()