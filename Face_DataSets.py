#OpenCV2 for image processing.
import cv2

#numpy for numerical and scientific computing.
import numpy as np
from pip._vendor.distlib.compat import raw_input

#"haarcascade_frontalface_default.xml" is pre-built frontal face training model from OpenCV using for Face detection.
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

#Starting Video Capturing and 0 indicates the default value for webcam.
cam = cv2.VideoCapture(0);

#id, sampleNum for labelling the dataSets.
id = raw_input("Enter your ID : ")
sampleNum = 0;

while(True):

    #Capturing Video Frame.
    ret, img = cam.read();

    #Converting the frame to grayScale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect frames of different sizes, list of faces rectangles.
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);

    #Loop for face detected in each frame.
    for(x,y,w,h) in faces :

        #incremented for each face(dataset) of a person.
        sampleNum = sampleNum + 1;

        #Saving the captured images into DataSet folder.
        cv2.imwrite("dataSet/User." + str(id) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

        #Creating rectanglar boundary around the detected face.
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

        #Time to each for capturing pictures.
        cv2.waitKey(100);

    #Display the video frame with the rectagle represented around the face above.
    cv2.imshow("Face", img);


    cv2.waitKey(1)

    #After saving 20 pictures, exit the loop.
    if(sampleNum > 20) :
        break;

#Stop video camera.
cam.release()

#Close all windows.
cv2.destroyAllWindows()