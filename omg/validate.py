# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:07:49 2018

@author: leopoldolusquino
"""
import wisardpkg as wp
import numpy
import scipy.io.wavfile
import cv2
import numpy as np
import csv
import os
from scipy.fftpack import dct
from math import sqrt
import ast
import preprocessing

pre_emphasis = 0.97
frame_size = 0.025
frame_stride = 0.01
NFFT = 512
nfilt = 40
num_ceps = 12
cep_lifter = 22
train_dataset = "../../omg_dataset/Training"
test_dataset = "../../omg_dataset/Test"
validation = "output"
results = "results"

class EmpathyPredictor(object):
	def __init__(self, model, n, generalized = False, min_score = 0.1, threshold = 10000, discriminator_limit = 6, min_zero = 0, min_one = 0, 
                       mean=wp.GeometricMean(), factor = 1, audio_type = "mfcc", integral_image = True, number_of_kernels = 1024,
                       bits_by_kernel = 3, activation_degree = 0.07, use_direction = True, face_type = "sauvola"):
		self.model = model

		if(model == "rew"):
		        self.net = wp.RegressionWisard(n, minZero = min_zero, minOne = min_one, mean = mean)
		if(model == "crew"):
			self.net = wp.ClusRegressionWisard(n, min_score, threshold, discriminator_limit, minZero = min_zero, minOne = min_one, mean = mean)
		
		self.n = n
		self.generalized = generalized
		self.min_score = min_score
		self.threshold = threshold
		self.discriminator_limit = discriminator_limit
		self.min_zero = min_zero
		self.min_one = min_one
		self.mean = mean
		self.factor = factor
		self.audio_type = audio_type
		self.integral_image = integral_image
		self.number_of_kernels = number_of_kernels
		self.bits_by_kernel = bits_by_kernel
		self.activation_degree = activation_degree
		self.use_direction = use_direction
		self.face_type = face_type
		self.audio_temp = 0
		self.audio_count = 0
	
	def get_preprocessing(self, audio, frame, face, face2 = ""):
		#print('entrou pre processing')
		if(self.audio_count == 0):
			audio = preprocessing.get_audio_binarized(audio, frame, input_type = self.audio_type, 
			integral_image = self.integral_image, number_of_kernels = self.number_of_kernels,
			bits_by_kernel = self.bits_by_kernel, activation_degree = self.activation_degree,
			use_direction = self.use_direction)
			self.audio_temp = audio
			self.audio_count += 1
		else:
			audio = self.audio_temp
			self.audio_count += 1
			if(self.audio_count == 60):
				self.audio_count = 0

		#print('pr audio')
		face = preprocessing.get_face_binarized(face, self.face_type, face2 = face2, self.generalized)

		return preprocessing.sum_bin_input(face, audio)	

	def build_dataset(folder, train = True):
		ds = wp.DataSet()

		dataset = ""
		if(train):
			dataset = train_dataset
		else:
			dataset = test_dataset

		for i in range(0, len(train_face_folders)):
			subject_folder = os.listdir(dataset + "/Faces/" + folder[i] + '/Subject/')
			for j in range(0, 1):
				face = dataset + '/Faces/' + folder[i] + '/Subject/'+ subject_folder[j]
				frame = int(subject_folder[i].replace(".png", ""))
				audio = dataset + '/audio/' + folder[i]+'.wav'
				valence = list(csv.reader(open(dataset + '/Annotations/' + folder[i].replace(".mp4", "") + 
								".csv")))[frame+1][0]
				valence = float(valence)
				if(self.generalized):
					ds.add(self.get_preprocessing(audio, frame, face, face2 = set + dataset + '/Faces/' + 
											folder[i] + '/Actor/'+ subject_folder[j]), valence)
				else:
					ds.add(self.get_preprocessing(audio, frame, face), valence)


	def iteration(self, train_name, test_name):
		train_face_folders = next(os.walk(train_dataset + "/Faces"))[1]
		ds_train = self.build_dataset(train_face_folders, True)
		ds_train.save(train_name)

		test_face_folders = next(os.walk(test_dataset + "/Faces"))[1]
		ds_test = self.build_dataset(test_face_folders, False)	
		ds_test.save(test_name)		

	def validate_rds(self, ds1_path, ds2_path, out):
		ds1 = hw.RegressionDataSet(ds1_path)
		ds2 = hw.RegressionDataSet(ds2_path)

		writer = open(out, "w+")
		self.net.train(ds1)
		for i in range(0, ds2.size()):
			bin = ds2.getInput( i )
			lista = [ bin.get( i ) for i in range(bin.size()) ]
			while(len(lista) != 35073):
				lista = [0] + lista
			result = self.net.predict([lista])
			writer.write(str(result[0]) + "\n")
		writer.close()

models = ["rew", "crew"]

for model in models:
    for i in range(10, 30, 5):
        print(str(i) + " " + model)
        ep = EmpathyPredictor(model, i)
        ep.validate_rds_total(ds1_path, ds2_path, model, i)
