import csv
import numpy as np
import os
import cv2
from celeb_data import grab_images

def load_data_wrapper(root, img_folder="Images", results_csv="Attributes/demographics/values.csv", img_width=178, img_height=218, limit=10000, gray_scale=True):
	values = get_expected_values(os.path.join(root, results_csv), limit=limit)
	names = list(values.keys())
	images = input_images(os.path.join(root, img_folder), filenames=names, limit=limit)

	training_inputs = []
	results = []
	for name in names:
		training_inputs.append(images[name])
		results.append(values[name])

	training_data = (np.array(training_inputs[0:int(limit/2)]), results[0:int(limit/2)])
	test_data = (np.array(training_inputs[int(limit/2)+1:]), results[int(limit/2)+1:limit])

	return training_data, test_data, None


def get_expected_values(path, limit=-1, column=14):
	with open(path, 'rU') as file:
		reader = csv.reader(file, dialect=csv.excel_tab)
		rows = []
		for row in reader:
			rows.append(row[0].split(","))
		values = []
		if (limit == -1):
			limit = len(rows)
		r = {}
		x = 0
		while len(r.keys()) < limit:
			x += 1
			m = rows[x][column]
			name = rows[x][0]
			if m == "NaN":
				r[name] = 0
			else:
				v = int(rows[x][column])
				if v == 1:
					r[name] = 0
				else:
					r[name] = 1

		return r

def input_images(folder, limit=-1, target_size=(178, 218), filenames=None):
	if filenames is None:
		filenames = os.listdir(folder)
		filenames.sort()
	imgs = {}
	for f in filenames[0:limit]:
		img = cv2.imread(os.path.join(folder, f))
		img = cv2.resize(img, target_size)
		imgs[f] = img
	return imgs