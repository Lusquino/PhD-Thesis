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

pre_emphasis = 0.97
frame_size = 0.025
frame_stride = 0.01
NFFT = 512
nfilt = 40
num_ceps = 12
cep_lifter = 22

def sum_bin_input(b1, b2):
    b = []
    for i in range(len(b1)):
        b.append(b1[i])
    for i in range(len(b2)):
        b.append(b2[i])
        
    return wp.BinInput(b)

def get_filter_banks_mfcc(audio, frame):
    ########### Setup ###############
    sample_rate, signal = scipy.io.wavfile.read(audio)  # File assumed to be in the same directory
    if(frame > 60):
        frame1 = frame - 60
    signal = signal[frame1:int(frame * sample_rate)]
    
    ########### Pre-Emphasis ##########
    
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    ############ Framing #############
    
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    
    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    
    ############# Window ###################
    
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    
    ############ Fourier-Transform and Power Spectrum #####################
    
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    ########### Filter Banks ###############
    
    low_freq_mel = 0 
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])	# left
        f_m = int(bin[m])			 # center
        f_m_plus = int(bin[m + 1])	# right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
    
    ############# Mel-frequency Cepstral Coefficients (MFCCs) ###########
    
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    
    (nframes, ncoeff) = mfcc.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift  #*
    
    ########### Mean Normalization ##############
    
    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    
    return filter_banks, mfcc

def integral_image_fn(cepstrum):
    for i in range(0, len(cepstrum)):
        total = 0
        for j in range(0, len(cepstrum[i])):
            cepstrum[i][j] += total
            total = cepstrum[i][j]
    
    return cepstrum

def append(v1, v2, dim_v1, dim_v2):
    v = [[0]* (dim_v1+dim_v2) for i in range(0, len(v1))]
    for i in range(0, len(v1)):
        for j in range(dim_v1):
            v[i][j] = v1[i][j]
        for j in range(dim_v2):
            v[i][dim_v1+j] = v2[i][j]
            
    return v

def get_audio_binarized(audio, frame, input_type = "mfcc", integral_image = True, number_of_kernels = 1024, 
                        bits_by_kernel = 3, activation_degree = 0.07, use_direction = True):
    if(input_type == "fb"):
        len_kernel = 40
    if(input_type == "mfcc"):
        len_kernel = 12
    if(input_type == "sum"):
        len_kernel = 52
        
    fb, mfcc = get_filter_banks_mfcc(audio, frame)
    
    if(input_type == "fb"):
        audio_input = fb
    if(input_type == "mfcc"):
        audio_input = mfcc
    if(input_type == "sum"):
        audio_input = append(fb, mfcc, 40, 12)

    if(integral_image == True):
        audio_input = integral_image_fn(audio_input)
        
    len_kernel /= 2
    
    audio_binarized = []

    l = []		
    for i in range(0, int(len(audio_input[0])/2)):
        l1 = []
        for j in range(0, len(audio_input)):
            l1.append([audio_input[j][2*i], audio_input[j][(2*i)+1]])
        l.append(l1)

    for i in range(0, len(l)):
        kernel = wp.KernelCanvas(2, number_of_kernels, bitsByKernel = bits_by_kernel, activationDegree = activation_degree, useDirection = use_direction)
        audio_binarized.append(kernel.transform(l[i]))
    
    return wp.BinInput(audio_binarized)

######### Faces #####################

def get_binary_input(image, threshold = 1.5):
    binary_input = []

    average_luminance = 0
    
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            color = image[x, y]
            average_luminance += get_luminance(color)
        
    average_luminance /= image.shape[0] * image.shape[1]

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            color = image[x, y]
            luminance = get_luminance(color)
            binary_input.append(1) if (luminance >= threshold * average_luminance) else binary_input.append(0)

    return binary_input

def calculate_histogram(image):
    histogram = [0]*256

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            color = image[x, y]
            histogram[int(get_luminance(color))] += 1
    return histogram

def calculate_mean(image):
    mean = 0
            
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            mean += sum(image[x, y])/3

    mean /= image.shape[0] * image.shape[1]
            
    return mean

def calculate_standard_deviation(histogram, mean):
    variance = 0

    for i in range(0, 256):
        variance += (histogram[i]-mean)**2

    return sqrt(variance/256)
    
def get_sauvola_binarization(image, weight = 1):
    binary_input = []
    
    histogram = calculate_histogram(image)
    mean = calculate_mean(image)
    standard_deviation = calculate_standard_deviation(histogram, mean)
    
    threshold = mean + weight * (standard_deviation/128 - 1) + 1
    
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            binary_input.append(1) if(sum(image[x, y])/3 > threshold) else binary_input.append(0)
            
    return binary_input

def get_luminance(color):
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]

def get_canny_filter(image, lower_bound = 4, upper_bound = 8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed_image = cv2.GaussianBlur(image, (7, 7), 0)

    canny_result = np.array(cv2.Canny(smoothed_image, lower_bound, upper_bound))
    binary_input = canny_result.reshape(canny_result.shape[0]*canny_result.shape[1])
    binary_input[binary_input>0] = 1 #255 values are converted to 1
    
    return binary_input.tolist()

def get_adaptive_gaussian(image):
    adaptive_result = np.array(cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,11,2))
    binary_input = adaptive_result.reshape(adaptive_result.shape[0]*adaptive_result.shape[1])
    binary_input[binary_input>0] = 1 #255 values are converted to 1
    
    return binary_input.tolist()

def get_otsu_binarization(image):
    smoothed_image = cv2.GaussianBlur(image,(5,5),0)
    otsu_result = cv2.threshold(smoothed_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary_input = otsu_result[1].reshape(otsu_result[1].shape[0] * otsu_result[1].shape[1]) 
    binary_input[binary_input>0] = 1 #255 values are converted to 1
    
    return binary_input.tolist()

def binarize_face(face, face_type):
    if((face_type == "sauvola") || (face_type == "canny")):
        face = cv2.imread(face1, cv2.IMREAD_COLOR)
    if((face_type == "gaussian") || (face_type == "otsu")):
        face = cv2.imread(face1, cv2.CV_8UC1)
    
    if(face_type == "sauvola"):
        face = get_sauvola_binarization(face1)
    if(face_type == "canny")
        face = get_canny_filter(face1)
    if(face_type == "gaussian")
        face =  get_adaptive_gaussian(face1)
    if(face_type == "otsu")
        face = get_otsu_binarization(face1)
    return face
    
def get_face_binarized(face1, face_type, face2 = "", generalized = False):
    face1 = wp.BinInput(binarize_face(face1, face_type))
        
    if(generalized):
        face2 = wp.BinInput(binarize_face(face2, face_type))
        return sum_bin_input(face1, face2)

    return face1