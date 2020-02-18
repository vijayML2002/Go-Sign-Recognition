import cv2
import numpy as np
import pickle
import serial
import time

ser1=serial.Serial('COM6',9600)

url = 'https://192.168.43.1:8080/video'
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('saved_reco.yml')

with open("label_data", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
print(labels)

cat=cv2.VideoCapture(0)

while True:
    _,frame=cat.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.3,minNeighbors=3)
    for (x,y,w,h) in faces:
        roi_gray=gray[x:x+w,y:y+h]
        id_, conf = face_recognizer.predict(roi_gray)
        if conf>0:
    	    print(id_)
    	    print(labels[id_])
    	    tick=time.time()
    	    if id_==0:ser1.write('0'.encode())
    	    tok=time.time()
    	    times=tok-tick
    	    print(times)
    	    cv2.putText(frame, labels[id_], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    	
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3);
    cv2.imshow('frame',frame)
    k=cv2.waitKey(1)
    if k==27:
        break
cat.release()
cv2.destroyAllWindows()
