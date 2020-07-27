import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
labels = {}
with open("labels.pickle", 'rb') as f:
	labels = pickle.load(f)
	labels = {v:k for k,v in labels.items()}#invert lookup

cap = cv2.VideoCapture(0)

boolin = True
count = 0 
while(boolin):


    #Capture frame
    ret, frame = cap.read()
    
    #to use this cascade to recognize faces the color must be in grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    


    for(x,y,w,h) in faces:     
        #to record the number of the trial
        count = count + 1
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycoord_start, ycoord_end)
        #roi_color =frame[y:y+h, x:x+w] 



        #recognizer
        id_, conf = recognizer.predict(roi_gray)
        if  conf <=53:
        	print( "|| Trial #: " +str(count) +" || Authorized as: " + labels[id_] + " || Confidence: " +str(conf)) 
        	
        	font = cv2.FONT_HERSHEY_SIMPLEX
        	name = labels[id_]
        	color = (255,255,255)
        	stroke = 2
        	cv2.putText(frame, "Authorized as: " + name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        else: 
        	print( "|| Trial #: " +str(count) +" ||Unauthorized User" + " || Confidence: " +str(conf))



        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255,0,0) #BGR
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke)

    #Display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(200) & 0xFF ==ord('q'):
        break
    


#Release
cap.release()
cv2.destroyAllWindows()
