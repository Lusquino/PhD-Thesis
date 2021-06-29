import glob
import numpy as np
from datetime import datetime
import os
from wisardpkg import DataSet, BinInput

def get_soil_data(filename='Data/soil_data.csv'):
    with open(filename) as f:
        lines = f.read().strip()
        lines = lines.split('\n')
        data = {}

        for i in range(1, len(lines)):
            line = lines[i].split(',')
            key = int(line[0])
            value = [int(j) for j in line[2:]]
            data[key] = value

    return data


def get_field_data(prefix='Data/field-', suffix='.csv'):
    data = []
    filenames = glob.glob("{}*{}".format(prefix, suffix))
    filenames.sort()

    for filename in filenames:
        with open(filename) as f:
            lines = f.read().strip()
            lines = lines.split('\n')
            file_data = {}

            for i in range(1,len(lines)):
                line = lines[i].split(',')
                key = (int(line[0]), int(line[1]))
                value = [float(j) for j in line[2:]]
                file_data[key] = value

        data.append(file_data)
    return data


def get_training_set(filename='Data/train.csv'):
    train_data = []
    train_target = []
    with open(filename) as f:
        lines = f.read().strip()
        lines = lines.split('\n')

        for i in range(1,len(lines)):
            line = lines[i].split(',')
            data = [int(j) for j in line[1:-1]]
            target = float(line[-1])

            train_data.append(data)
            train_target.append(target)

    return train_data, train_target


def get_test_set(filename='Data/test.csv'):
    with open(filename) as f:
        lines = f.read().strip()
        lines = lines.split('\n')
        data = []

        for i in range(1,len(lines)):
            line = lines[i].split(',')
            data.append([int(j) for j in line[1:]])

    return data

def get_datasets(filename='Data/train.csv'):
    train_dataset = DataSet()
    test_dataset = DataSet()
    
    thermometer = 10
    window = 20
    
    X_train, Y_train = get_training_set()
    X_test = get_test_set()
      
    fieldData = get_field_data()
    r = 1 + 8 * window
    nBits = [thermometer for i in range(r)]
    #trData, tsData = join_data(X_train, X_test, fieldData, window, nBits)
    trData, tsData, trField, tsField = join_data(X_train, X_test, fieldData, window, nBits)
    X = np.array(trData)
    X_ = np.array(tsData)
    Y = np.array(Y_train)
    
    for i in range(X.shape[0]):
        train_dataset.add(BinInput(X[i]), Y[i])
    
    for i in range(X_.shape[0]):
        test_dataset.add(BinInput(X_[i]), Y[i])
        
    train_dataset.save("DS_train")
    test_dataset.save("DS_test")

def month_sub(month, year, sub_month):
    result_month = 0
    result_year = 0
    if month > (sub_month % 12):
        result_month = month - (sub_month % 12)
        result_year = year - int(sub_month / 12)
    else:
        result_month = 12 - (sub_month % 12) + month
        result_year = year - int(sub_month / 12 + 1)
    return (result_month, result_year)


def normalize(data):
    result = np.array(data)
    data_max = [np.max(result[:,i]) for i in range(result.shape[1])]
    data_min = [np.min(result[:,i]) for i in range(result.shape[1])]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if data_max[j]-data_min[j] != 0:
                result[i,j] = (result[i,j]-data_min[j]) / (data_max[j]-data_min[j])
            else:
                result[i,j] = (result[i,j]-data_min[j])

    return result

def thermometerEncoding(value, size):
    delta = 1. / size
    lims = [0]

    for i in range(1, size):
        lims.append(lims[i-1] + delta)

    word = []

    for i in range(size):
        if value > lims[i]:
            word.append(1)
        else:
            word.append(0)

    return word


def binarize(set, n_bits):
    bin = []
    for i in range(set.shape[0]):
        line = []
        for j in range(set.shape[1]):
            line += thermometerEncoding(set[i,j], n_bits[j])

        bin.append(line)

    return bin


def create_output_file(out, output_path = "out"):
    text = 'Id,production\n'
    
    for i in range(5243, 9353):
        text += str(i) + ',' + str(out[i-5243]) + '\n'

    #output_path = "out"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open("{}/submission{}.csv".format(output_path, datetime.now().strftime('%m_%d-%H-%M-%S')), 'w') as f:
        f.write(text.strip())


def join_data(trainData, testData, fieldData, timeWindow, nBits):
    Xtrain = []
    Xtest = []
    Ftrain = []
    Ftest = []

    for sample in trainData:
        field = sample[0]
        age = sample[1]
        # palmType = sample[2]
        harvestYear = int(sample[3])
        harvestMonth = int(sample[4])

        fd = []

        for i in range(1, timeWindow + 1):
            d = month_sub(harvestMonth, harvestYear, i)
            fd += fieldData[field][d]

        Xtrain.append([age] + fd)
        Ftrain.append(str(field))

    for sample in testData:
        field = sample[0]
        age = sample[1]
        # palmType = sample[2]
        harvestYear = int(sample[3])
        harvestMonth = int(sample[4])

        fd = []

        for i in range(1, timeWindow + 1):
            d = month_sub(harvestMonth, harvestYear, i)
            fd += fieldData[field][d]

        # if palmType == -1 or palmType == 7:
        #     palmType = 5

        Xtest.append([age] + fd)
        Ftest.append(str(field))

    normTrainData = normalize(Xtrain)
    normTestData = normalize(Xtest)

    binTrainData = binarize(normTrainData, nBits)
    binTestData = binarize(normTestData, nBits)

    return binTrainData, binTestData, Ftrain, Ftest
