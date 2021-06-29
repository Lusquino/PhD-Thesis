import glob
import os
import pickle
from pathlib import Path

import numpy as np
from skimage.filters import threshold_sauvola

import cv2


class CKP:
    def __init__(self, directory="data", preprocessing=None, labels_dir='FACS/', imgs_dir='cohn-kanade-images/', landmarks_dir='Landmarks/', remove_intensity=True):
        self.labelsType = ""
        self.remove_intensity = remove_intensity
        if labels_dir == 'FACS/':
            self.labelsType = "FACS"
        else:
            self.labelsType = "Emotion"

        self.directory = directory
        self.preprocessing = preprocessing

        Path(self.folder).mkdir(parents=True, exist_ok=True)

        outputFile = "ckp"

        #print("OK")
        transformedData = False
        if self.preprocessing != None:
            if os.path.isfile(self.folder + "{}{}{}.pkl".format(outputFile, self.labelsType, str(self.preprocessing))):
                #print("ENTREI 0")
                self.load("{}{}{}".format(
                    outputFile, self.labelsType, str(self.preprocessing)))
                transformedData = True
        if not transformedData:
            #print("ENTREI 1")
            if os.path.isfile(self.folder + "{}{}.pkl".format(outputFile, self.labelsType)):
                #print("ENTREI 2")
                self.load("{}{}".format(outputFile, self.labelsType))
            else:
                #print("ENTREI 3")
                self.setData(self.folder + labels_dir,
                             self.folder + imgs_dir,
                             self.folder + landmarks_dir)
                self.save("{}{}".format(outputFile, self.labelsType))

            if self.preprocessing != None:
                #print("ENTREI 4")
                self.data = self.preprocessing.transform(self.data.copy())
                self.save("{}{}{}".format(outputFile, self.labelsType, str(self.preprocessing)))
        
    def setData(self, labels_dir, imgs_dir, landmarks_dir):
        self.data = []
        self.labels = []
        len_text = 0
        for filename in glob.glob("{}**/**/*.txt".format(labels_dir)):
            img_path = Path(
                imgs_dir + filename[len(labels_dir):-len(self.labelsType)-5] + '.png')

            if img_path.is_file():
                with open(str(filename), 'r') as myfile:
                    imgLabels = [[], []]
                    count = 1
                    for elem in myfile.read().replace('\n', '').split(" "):
                        if elem != '':
                            imgLabels[count % 2].append(str(int(float(elem))))
                        count += 1
                    
                    if len(self.labelsType) == 4:
                        if self.remove_intensity:
                            self.labels.append(imgLabels[0])
                        else:
                            self.labels.append(imgLabels)
                    else:
                        self.labels.append(imgLabels[0][0])

                filelandmarks = Path(
                    landmarks_dir + filename[len(labels_dir):-len(self.labelsType)-5] + '_landmarks.txt')
                print(filename, end='\r')
                len_text = len(filename)
                img = self.applyLandmarks(str(img_path), str(filelandmarks))
                self.data.append(np.ravel(img))
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(" "*len_text, end='\r')

    def loadLandmarks(self, filename):
        landmarks = []
        with open(filename) as f:
            for line in f.readlines():
                x = int(float(line.split()[0]))
                y = int(float(line.split()[1]))
                landmarks.append((x, y))
            f.close()
        return landmarks

    def applyLandmarks(self, filename, filelandmarks):
        image = cv2.imread(filename, 0)
        landmarks = self.loadLandmarks(filelandmarks)
        rect = [-1, -1, -1, -1]
        for i, (x, y) in enumerate(landmarks):
            if x < rect[0] or rect[0] == -1:
                rect[0] = x
            if x > rect[2] or rect[2] == -1:
                rect[2] = x
            if y < rect[1] or rect[1] == -1:
                rect[1] = y
            if y > rect[3] or rect[3] == -1:
                rect[3] = y

        new_img = image[rect[1]:rect[3], rect[0]:rect[2]]
        new_img = cv2.resize(new_img, (100, 100))

        return new_img

    @property
    def folder(self):
        return os.path.join(self.directory, self.__class__.__name__) + '/'

    def save(self, filename="ckp"):
        with open(self.folder + filename + ".pkl", 'wb') as f:
            pickle.dump([self.data, self.labels], f)

    def load(self, filename="ckp"):
        with open(self.folder + filename + ".pkl", 'rb') as f:
            self.data, self.labels = pickle.load(f)
            print(f)
            print("LOAD")
            print(self.data)

class AdaptativeMean:
    def __init__(self, blockSize=11, c=2):
        self.blockSize = blockSize
        self.c = c

    def __str__(self):
        return "{}_{}-{}".format(self.__class__.__name__, self.blockSize, self.c)

    def transform(self, data):
        for i in range(len(data)):
            data[i] = np.ravel(cv2.adaptiveThreshold(
                np.reshape(data[i], (-1, 100)), 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.blockSize, self.c))
        return data


class AdaptativeGaussian:
    def __init__(self, blockSize=11, c=2):
        self.blockSize = blockSize
        self.c = c

    def __str__(self):
        return "{}_{}-{}".format(self.__class__.__name__, self.blockSize, self.c)

    def transform(self, data):
        for i in range(len(data)):
            data[i] = np.ravel(cv2.adaptiveThreshold(
                np.reshape(data[i], (-1, 100)), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.blockSize, self.c))
        return data


class Sauvola:
    def __init__(self, window_size=25, k=0.2):
        self.window_size = window_size
        self.k = k

    def __str__(self):
        return "{}_{}-{}".format(self.__class__.__name__, self.window_size, self.k)

    def transform(self, data):
        for i in range(len(data)):
            n_data = np.reshape(data[i], (-1, 100))
            thresh_sauvola = threshold_sauvola(n_data, window_size=self.window_size, k=self.k)
            data[i] = np.array(n_data > thresh_sauvola).flatten()
        return data

if __name__ == "__main__":
    dataset = CKP(preprocessing=Sauvola())

    cv2.imwrite("test.png", np.reshape(dataset.data[2], (-1, 100))*255)

