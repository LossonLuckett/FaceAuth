import os
import cv2
import numpy as np
import pickle
from PIL import Image

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = 	os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Images")

current_id = 0
label_ids = {}
y_labels = []
x_train = []

#This section of code is used to pull labels from the images in the Image folder, and converts the grayscale image into a numpy array which will be used to train our model
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jfif") or file.endswith("jpeg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			#print(path, label)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			print(label_ids)
			#y_labels.append(label)
			#x_train.append(path)
			pil_image = Image.open(path).convert("L") #Converted Image into grey scale
			size = (550, 550) 
			final_image = pil_image.resize(size, Image.ANTIALIAS) #Final resized greyscale image
			#resize images for training purposes
			image_array = np.array(final_image, "uint8") #convert the image to a NUMPY Array 
			print (image_array)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)# Running the face detection on the image array


			#Grab the region of interest of the image array	
			for (x,y,w,h) in faces:
				roi = image_array[y:y+ h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)
#print(y_labels)
#print(x_train) 

with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)


print(y_labels)
recognizer.train(x_train, np.array(y_labels)) #Train the recognizer
recognizer.save("trainer.yml") #Save the trained recognizer into a .yml file
