import sys, traceback
import cv2
import numpy as np
import matplotlib.pyplot as plt
import inspect

from os import walk
from skimage.feature import hog
from skimage import data, color, exposure

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

def trainning_classifier(desc_SVM_input, classInput):
    print '------- Trainning Classifier ----------'
    trainned_classifier_list = []
    counter = 8
    
    for corpus_name in desc_SVM_input.keys():
        clf2 = GaussianNB()
        clf2.fit(desc_SVM_input[corpus_name], classInput)
        clf = svm.SVC(kernel='linear', probability=True)
        clf.fit(desc_SVM_input[corpus_name], classInput)
        trainned_classifier_list.append([corpus_name, clf2])
        counter += 1
    return trainned_classifier_list

def testing_classifier(trainned_classifier_list, desc_SVM_test, classTest):
    files_number = 0
    counter = 0
    for test_corpus in desc_SVM_test.keys():
        print test_corpus
#       print desc_SVM_test[test_corpus]
        print trainned_classifier_list[counter][1].predict_proba(desc_SVM_test[test_corpus])
        print trainned_classifier_list[counter][1].predict(desc_SVM_test[test_corpus])
        print accuracy_score(trainned_classifier_list[counter][1].predict(desc_SVM_test[test_corpus]),classTest)
        counter += 1