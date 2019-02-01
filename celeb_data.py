import csv
import cv2
import os
import numpy as np

def load_data_wrapper(root, img_folder="img_align_celeba", results_csv="list_attr_celeba.csv", img_width=178, img_height=218, limit=10000, gray_scale=True):
	img_area = img_width * img_height
	training_inputs = get_input_images(os.path.join(root, img_folder), limit=limit, gray_scale=gray_scale)
	# training_inputs = [x.reshape((img_area, 1)) for x in training_inputs]
	results = get_expected_values(os.path.join(root, results_csv), limit=limit)

	# cv2.imshow('preview', images[0])

	training_data = (np.array(training_inputs[0:int(limit/2)]), results[0:int(limit/2)])
	test_data = (np.array(training_inputs[int(limit/2)+1:]), results[int(limit/2)+1:limit])
	# training_data = zip(training_inputs, results)

	return training_data, test_data, None

def get_expected_values(path, limit=-1, column=21):
	with open(path, 'r') as file:
		reader = csv.reader(file)
		values = [row[column] for row in reader]

		print("Attr: " + values[0])
		r = np.empty((limit))
		for x in range(1, limit + 1):
			v = int(values[x])
			if v == 1:
				r[x - 1] = 0
			else:
				r[x - 1] = 1
		print(r)

		return r

def grab_images(folder, limit=-1):
	filenames = os.listdir(folder)
	filenames.sort()
	imgs = [cv2.imread(os.path.join(folder, f)) for f in filenames[0:limit]]

def get_input_images(folder, limit=-1, gray_scale=True):
	imgs = grab_images(folder, limit=limit)
	if (gray_scale):
		imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
	return np.array(imgs)