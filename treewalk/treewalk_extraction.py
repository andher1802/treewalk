import sys, traceback
import inspect

import cv2
import numpy as np
import matplotlib.pyplot as plt

from os import walk

from skimage.feature import hog
from skimage import data, color, exposure

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

def getSiftDescriptors(image, imagetoprocess, classValidator, draw):
	gray = cv2.cvtColor(imagetoprocess,cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, desc = sift.detectAndCompute(np.array(imagetoprocess), None)
	if draw:
		siftImg=cv2.drawKeypoints(imagetoprocess,kp,np.array(image))
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
		ax1.axis('off')
		ax1.imshow(image, cmap=plt.cm.gray)
		ax1.set_title('ORIGINAL')
		ax1.set_adjustable('box-forced')
		ax2.axis('off')
		ax2.imshow(siftImg, cmap=plt.cm.gray)
		ax2.set_title('SIFT DESCRIPTOR')
		ax2.set_adjustable('box-forced')
		plt.show()
	return kp, desc, classValidator

def getHogDescriptor(image, imagetoprocess, classValidator, draw):
	gray2 = color.rgb2gray(imagetoprocess)
	fd, hog_image = hog(gray2, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
	if draw:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
		ax1.axis('off')
		ax1.imshow(image, cmap=plt.cm.gray)
		ax1.set_title('ORIGINAL')
		ax1.set_adjustable('box-forced')
		ax2.axis('off')
		ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
		ax2.set_title('HOG DESCRIPTOR')
		ax2.set_adjustable('box-forced')
		plt.show()
	return fd.reshape(-1,1)

def getColorDescriptor(image, imagetoprocess, component, draw):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	channels = cv2.split(hsv)[component]
	hist = exposure.histogram(channels)
	norm_hist = [(element*(10**6))/hist[0].sum() for element in hist[0]]
	if len(norm_hist)<255:
		for i in range(255-len(norm_hist)):
			norm_hist.append(0)
	if draw:
		fig, (ax1) = plt.subplots(1, figsize=(20, 10), sharex=True, sharey=True)
		ax1.hist(norm_hist,range(256))
		ax1.set_title('COLOR DESCRIPTOR')
		ax1.set_adjustable('box-forced')
		plt.show()
	dummy_result = np.array(norm_hist).reshape(-1,1)
	return dummy_result

def getBOW(dirpath, files_number, features = ['sift', 'hog'], cluster_numbers = {'sift':10, 'hog':20}):
	filePDDI = []
	for dirpath, dirname, filename in walk(dirpath):
		filePDDI.extend(filename)

	filePDDI = [name for name in filePDDI if name.split('.')[-1] == 'jpg']
	filestocompute = []
	
	if files_number != 0: 
		filestocompute = list(filePDDI[:files_number])
	else:
		filestocompute = list(filePDDI)

	print '------- creating Descriptor ----------'
	classInput = []

	bag_of_words_desc = {}
	hist_desc = {}

	for name in features:
		bag_of_words_desc[name] = []
		hist_desc[name] = []

	for filename in filestocompute:
		print filename
		classInput.append(filename.split('_')[0])
		file = dirpath + filename
		img = cv2.imread(file)
		imagetoprocess = img
		for feature in features:
			if feature == 'sift':
				kp, desc, classValidator = getSiftDescriptors(img, imagetoprocess, filename.split('_')[0], False)
				bag_of_words_desc[feature].extend(desc)
				hist_desc[feature].append(desc)
			elif feature == 'hog':
				fd = getHogDescriptor(img, imagetoprocess, filename.split('_')[0], False)
				bag_of_words_desc[feature].extend(fd)
				hist_desc[feature].append(fd)
			elif feature == 'color': 
				hsv = cv2.cvtColor(imagetoprocess, cv2.COLOR_BGR2HSV)
				component = 0
				color = getColorDescriptor(img, hsv, component, False)
				bag_of_words_desc[feature].extend(color)
				hist_desc[feature].append(color)

	desc_names = features
	cluster_result = {}

	print '------- creating Clusters ----------'
	for name in desc_names:
		if name in bag_of_words_desc.keys():
			cluster_result[name] = KMeans(n_clusters=cluster_numbers[name], random_state=0).fit(bag_of_words_desc[name])

	print '------- creating BOW ----------'
	desc_SVM_input = {}
	desc_counter = 0
	for name in desc_names:
		if name in hist_desc.keys():
			desc_SVM_input[name] = [] 
			start_point = 0
			for file_sift in hist_desc[name]:
				slide_length = len(file_sift)
				slide_cluster = cluster_result[name].labels_[start_point:start_point+slide_length]
				image_hist_cluster = [0 for element in range(cluster_numbers[name])]
				total_labels = 0
				for label in slide_cluster:
					start_point += 1
					image_hist_cluster[label] += 1
					total_labels += 1 
				image_hist_cluster = [element/(total_labels*1.00) for element in image_hist_cluster]
				desc_SVM_input[name].append(image_hist_cluster)
			desc_counter += 1
	return desc_SVM_input, classInput

def extract_features(dirpath):
	files_number = 0
	features = ['sift','hog','color']
	clusters_number = {'sift':50, 'hog':50, 'color':20}
	desc_SVM_input, classInput  = getBOW(dirpath, files_number, features, clusters_number)
	return desc_SVM_input, classInput