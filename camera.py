import tensorflow as tf
import cv2
import time
import numpy as np
import math

model_file="current-model"
genderClassifer = tf.keras.models.load_model(model_file)

faceCascade = cv2.CascadeClassifier("face_detection.xml")

capture = cv2.VideoCapture(0)
window_name = "video"
face_window_name = "face"



cv2.namedWindow(window_name, 0)
cv2.resizeWindow(window_name, 600, 600)

cv2.namedWindow(face_window_name, 0)
cv2.resizeWindow(face_window_name, 178, 218)

def getFaces(img, classifer):
	return classifer.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

def getImgForGenderClassification(img, shape):
	## Classifer works best if there is area around the face
	## increase shape
	final_shape = (178, 218)
	if final_shape[0] > shape[0] or final_shape[1] > shape[1]:
		## The face is too far away
		return None

	# (x,y,w,h) = shape
	# shape = (max(x - 50, 0), max(y - 50, 0), w+50, h+50)
	face_img = img[shape[1]:shape[1] + shape[3], shape[0]:shape[0] + shape[3]]

	# (w,h,c) = face_img.shape
	# if (w == 0 or h == 0):
	# 	print("Failed to get face image")
	# 	return None
	return cv2.resize(face_img, final_shape)



while(capture.isOpened()):
	ret, frame = capture.read()
	
	if ret == True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = getFaces(gray, faceCascade)
		text_placement = 100
		for shape in faces:
			(x,y,w,h) = shape
			face = getImgForGenderClassification(frame, shape)
			if face is not None:
				prediction = genderClassifer.predict(np.array([face]))
				classification = prediction.argmax()
				accuracy = prediction[0][classification]
				accuracy = math.floor(100*accuracy)
				gender = "Female"
				print(accuracy)
				if (classification == 0):
					gender="Male"
				cv2.putText(frame, gender, (150, text_placement), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0))
				cv2.putText(frame, "%i%%" % accuracy, (0, text_placement), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0))
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				cv2.imshow(face_window_name, face)
				text_placement += 50
		cv2.imshow(window_name, frame)
		cv2.waitKey(1)
	else:
		break
