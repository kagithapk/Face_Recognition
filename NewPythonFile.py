#OpenCV2 for image processing.
import cv2

#numpy for numerical and scientific computing.
import numpy as np

#"haarcascade_frontalface_default.xml" is pre-built frontal face training model from OpenCV using for Face detection.
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

#Starting Video Capturing and 0 indicates the default value for webcam.
cam = cv2.VideoCapture(0);

#Loop to detect the face i.e.,rectangular box around the detected face on the screen capturing from Video Camera.
while(True):

    #Capturing Video Frame.
    ret, img = cam.read();

    #Converting the frame to grayScale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect frames of different sizes, list of faces rectangles.
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);

    #Loop for face detected in each frame.
    for(x,y,w,h) in faces :

        #Creating rectanglar boundary around the detected face.
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

    #Display the video frame with the rectagle represented around the face above.
    cv2.imshow("Face", img);

    #Press 'q' to quit the window representing the video.
    if(cv2.waitKey(1) == ord('q')) :
        break;

#Release the video camera.
cam.release()

#Closing all the windows.
cv2.destroyAllWindows()