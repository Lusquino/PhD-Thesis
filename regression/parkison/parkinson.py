#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

#Parkinson
print('-- STARTING : Parkinson (motor) --')

#load dataset
fileName = 'Parkinson/parkinsons_updrs.data'
data = pd.read_csv(fileName,sep=',')
print('Size:', data.shape)

#categorical data transformation
categorical_cols = ['sex']
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(data[categorical_cols]))
OH_cols.index = data.index
num_X = data.drop(categorical_cols, axis=1)
data = pd.concat([num_X, OH_cols], axis=1)
print('New size:',data.shape)

#remove irrelevant collumns
data = data.drop('subject#', axis=1)
data = data.drop('total_UPDRS', axis=1)
print('Final size:',data.shape)

data.head()

X = data[data.columns.difference(['motor_UPDRS'])]
y = data['motor_UPDRS']

import wisardpkg as wsd
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error

def allDividers(value):
    div = []
    for i in range(2, int(value/2) + 1):
        if value % i == 0:
            div.append(i)
    div.append(value)
    return div

def saveToFile(fileName, values):
    with open(fileName, 'a+') as f:
        f.write(str(values) + '\n')

def RegressionWiSARDExperiment(X_train, y_train, X_test, y_test):
    resultsP = []
    resultsV = []
    resultsT = []

    #preprocessing
    means = [wsd.PowerMean(2), wsd.HarmonicMean(), wsd.HarmonicPowerMean(2), wsd.GeometricMean(), wsd.ExponentialMean(), wsd.SimpleMean(),wsd.Median()]

    mins = [ np.min(X_train[:,i]) for i in range(X_train.shape[1]) ]
    maxs = [ np.max(X_train[:,i]) for i in range(X_train.shape[1]) ]

    for t in range(5,31):
        therm = [t for i in range(X_train.shape[1])]
        sTherm = np.sum(therm)

        DT = wsd.DynamicThermometer(therm, mins, maxs)

        binX_train = [DT.transform(X_train[i]) for i in range(X_train.shape[0])]
        binX_test = [DT.transform(X_test[i]) for i in range(X_test.shape[0])]

        DS_train = wsd.DataSet(binX_train,y_train)
        DS_test = wsd.DataSet(binX_test,y_test)

        for addr in range(int(np.sqrt(sTherm)), int(sTherm / 2) + 1):
            for m in means:
                maetr = []
                maets = []
                msetr = []
                msets = []
                t1 = []
                t2 = []
                t3 = []
                for i in range(10):
                    print(i,m,addr,t)
                    rwsd = wsd.RegressionWisard(addressSize=addr, mean=m, completeAddressing=True)
                    a = time.perf_counter()
                    rwsd.train(DS_train)
                    b = time.perf_counter()
                    delta1 = b - a
                    a = time.perf_counter()
                    outTrain = np.array(rwsd.predict(DS_train))
                    b = time.perf_counter()
                    delta2 = b - a
                    a = time.perf_counter()
                    outTest = np.array(rwsd.predict(DS_test))
                    b = time.perf_counter()
                    delta3 = b - a

                    outTrain[np.isnan(outTrain)] = np.nanmean(outTrain)
                    outTest[np.isnan(outTest)] = np.nanmean(outTest)
                    outTrain[outTrain==np.inf] = 0
                    outTest[outTest==np.inf] = 0

                    maetr.append(mean_absolute_error(outTrain,y_train))
                    maets.append(mean_absolute_error(outTest,y_test))
                    msetr.append(mean_squared_error(outTrain,y_train))
                    msets.append(mean_squared_error(outTest,y_test))

                    t1.append(delta1)
                    t2.append(delta2)
                    t3.append(delta3)

                resultsP.append([t, addr, m])
                resultsV.append([np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets)])
                resultsT.append([np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])

                saveToFile('housepriceRew.csv', [t, addr, m, np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets), np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])

    resultsP = np.array(resultsP)
    resultsV = np.array(resultsV)
    resultsT = np.array(resultsT)

    saveToFile('housepriceRew.csv',[resultsP[np.argmin(resultsV[:,2])], resultsV[np.argmin(resultsV[:,2])], resultsT[np.argmin(resultsV[:,2])]])
    saveToFile('housepriceRew.csv',[resultsP[np.argmin(resultsV[:,6])], resultsV[np.argmin(resultsV[:,6])], resultsT[np.argmin(resultsV[:,6])]])

def NTupleRegressorExperiment(X_train, y_train, X_test, y_test):
    resultsP = []
    resultsV = []
    resultsT = []

    #preprocessing

    possibleAddr = {}
    means = [wsd.SimpleMean()]

    mins = [ np.min(X_train[:,i]) for i in range(X_train.shape[1]) ]
    maxs = [ np.max(X_train[:,i]) for i in range(X_train.shape[1]) ]

    for t in range(5,31):
        therm = [t for i in range(X_train.shape[1])]
        sTherm = np.sum(therm)

        DT = wsd.DynamicThermometer(therm, mins, maxs)

        binX_train = [DT.transform(X_train[i]) for i in range(X_train.shape[0])]
        binX_test = [DT.transform(X_test[i]) for i in range(X_test.shape[0])]

        DS_train = wsd.DataSet(binX_train,y_train)
        DS_test = wsd.DataSet(binX_test,y_test)

        for addr in range(int(np.sqrt(sTherm)), int(sTherm / 2) + 1):
            for m in means:
                maetr = []
                maets = []
                msetr = []
                msets = []
                t1 = []
                t2 = []
                t3 = []
                for i in range(1):
                    print(i,m,addr,t)
                    rwsd = wsd.RegressionWisard(addressSize=addr, mean=m, orderedMapping=True, completeAddressing=True)
                    a = time.perf_counter()
                    rwsd.train(DS_train)
                    b = time.perf_counter()
                    delta1 = b - a
                    a = time.perf_counter()
                    outTrain = np.array(rwsd.predict(DS_train))
                    b = time.perf_counter()
                    delta2 = b - a
                    a = time.perf_counter()
                    outTest = np.array(rwsd.predict(DS_test))
                    b = time.perf_counter()
                    delta3 = b - a

                    outTrain[np.isnan(outTrain)] = np.nanmean(outTrain)
                    outTest[np.isnan(outTest)] = np.nanmean(outTest)
                    outTrain[outTrain==np.inf] = 0
                    outTest[outTest==np.inf] = 0

                    maetr.append(mean_absolute_error(outTrain,y_train))
                    maets.append(mean_absolute_error(outTest,y_test))
                    msetr.append(mean_squared_error(outTrain,y_train))
                    msets.append(mean_squared_error(outTest,y_test))

                    t1.append(delta1)
                    t2.append(delta2)
                    t3.append(delta3)

                resultsP.append([t, addr, m])
                resultsV.append([np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets)])
                resultsT.append([np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])

                saveToFile('housepriceNTuple.csv', [t, addr, m, np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets), np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])

    resultsP = np.array(resultsP)
    resultsV = np.array(resultsV)
    resultsT = np.array(resultsT)

    saveToFile('housepriceNTuple.csv',[resultsP[np.argmin(resultsV[:,2])], resultsV[np.argmin(resultsV[:,2])], resultsT[np.argmin(resultsV[:,2])]])
    saveToFile('housepriceNTuple.csv',[resultsP[np.argmin(resultsV[:,6])], resultsV[np.argmin(resultsV[:,6])], resultsT[np.argmin(resultsV[:,6])]])

def DecitionTreeExperiment(X_train, y_train, X_test, y_test):
    
    tree_model = DecisionTreeRegressor()
    a = time.perf_counter()
    tree_model.fit(X_train, y_train)
    b = time.perf_counter()
    delta0 = b - a
    a = time.perf_counter()
    tree_preds_train = tree_model.predict(X_train)
    b = time.perf_counter()
    delta1 = b - a
    a = time.perf_counter()
    tree_preds_test = tree_model.predict(X_test)
    b = time.perf_counter()
    delta2 = b - a

    saveToFile('housepriceDT.csv', [mean_absolute_error(y_train,tree_preds_train), mean_squared_error(y_train,tree_preds_train), mean_absolute_error(y_test,tree_preds_test), mean_squared_error(y_test,tree_preds_test), delta0, delta1, delta2])

def RandomForestExperiment(X_train, y_train, X_test, y_test):

    forest_model = RandomForestRegressor()
    a = time.perf_counter()
    forest_model.fit(X_train, y_train)
    b = time.perf_counter()
    delta0 = b - a
    a = time.perf_counter()
    forest_preds_train = forest_model.predict(X_train)
    b = time.perf_counter()
    delta1 = b - a
    a = time.perf_counter()
    forest_preds_test = forest_model.predict(X_test)
    b = time.perf_counter()
    delta2 = b - a

    saveToFile('housepriceRF.csv', [mean_absolute_error(y_train,forest_preds_train), mean_squared_error(y_train,forest_preds_train), mean_absolute_error(y_test,forest_preds_test), mean_squared_error(y_test,forest_preds_test), delta0, delta1, delta2])

def XGBoostExperiment(X_train, y_train, X_test, y_test):

    xgb_model = XGBRegressor()
    a = time.perf_counter()
    xgb_model.fit(X_train, y_train)
    b = time.perf_counter()
    delta0 = b - a
    a = time.perf_counter()
    xgb_preds_train = xgb_model.predict(X_train)
    b = time.perf_counter()
    delta1 = b - a
    a = time.perf_counter()
    xgb_preds_test = xgb_model.predict(X_test)
    b = time.perf_counter()
    delta2 = b - a

    saveToFile('housepriceXGB.csv', [mean_absolute_error(y_train,xgb_preds_train), mean_squared_error(y_train,xgb_preds_train), mean_absolute_error(y_test,xgb_preds_test), mean_squared_error(y_test,xgb_preds_test), delta0, delta1, delta2])

def ClusRegressionWiSARDExperiment(X_train, y_train, X_test, y_test):
    resultsP = []
    resultsV = []
    resultsT = []

    #preprocessing

    possibleAddr = {}
    means = [wsd.PowerMean(2), wsd.HarmonicMean(), wsd.HarmonicPowerMean(2), wsd.GeometricMean(), wsd.ExponentialMean(), wsd.SimpleMean(),wsd.Median()]

    mins = [ np.min(X_train[:,i]) for i in range(X_train.shape[1]) ]
    maxs = [ np.max(X_train[:,i]) for i in range(X_train.shape[1]) ]

    for t in range(5,31):
        therm = [t for i in range(X_train.shape[1])]
        sTherm = np.sum(therm)

        DT = wsd.DynamicThermometer(therm, mins, maxs)

        binX_train = [DT.transform(X_train[i]) for i in range(X_train.shape[0])]
        binX_test = [DT.transform(X_test[i]) for i in range(X_test.shape[0])]

        DS_train = wsd.DataSet(binX_train,y_train)
        DS_test = wsd.DataSet(binX_test,y_test)

        for addr in range(int(np.sqrt(sTherm)), int(sTherm / 2) + 1):
            for m in means:
                maetr = []
                maets = []
                msetr = []
                msets = []
                t1 = []
                t2 = []
                t3 = []
                for ms in np.arange(0.1, 1, 0.1):
                    for threshold in range(100,1001,100):
                        for limit in range(2,7):
                            for i in range(10):
                                print(i,m,addr,t)
                                rwsd = wsd.ClusRegressionWisard(addressSize=addr, mean=m, completeAddressing=True, minScore=ms, threshold=threshold, limit=limit)
                                a = time.perf_counter()
                                rwsd.train(DS_train)
                                b = time.perf_counter()
                                delta1 = b - a
                                a = time.perf_counter()
                                outTrain = np.array(rwsd.predict(DS_train))
                                b = time.perf_counter()
                                delta2 = b - a
                                a = time.perf_counter()
                                outTest = np.array(rwsd.predict(DS_test))
                                b = time.perf_counter()
                                delta3 = b - a

                                outTrain[np.isnan(outTrain)] = np.nanmean(outTrain)
                                outTest[np.isnan(outTest)] = np.nanmean(outTest)
                                outTrain[outTrain==np.inf] = 0
                                outTest[outTest==np.inf] = 0

                                maetr.append(mean_absolute_error(outTrain,y_train))
                                maets.append(mean_absolute_error(outTest,y_test))
                                msetr.append(mean_squared_error(outTrain,y_train))
                                msets.append(mean_squared_error(outTest,y_test))

                                t1.append(delta1)
                                t2.append(delta2)
                                t3.append(delta3)

                resultsP.append([t, addr, m, ms, threshold, limit])
                resultsV.append([np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets)])
                resultsT.append([np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])

                saveToFile('parkinsonCrew.csv', [t, addr, m, ms, threshold, limit, np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets), np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])

    resultsP = np.array(resultsP)
    resultsV = np.array(resultsV)
    resultsT = np.array(resultsT)

    saveToFile('parkinsonCrew.csv',[resultsP[np.argmin(resultsV[:,2])], resultsV[np.argmin(resultsV[:,2])], resultsT[np.argmin(resultsV[:,2])]])
    saveToFile('parkinsonCrew.csv',[resultsP[np.argmin(resultsV[:,6])], resultsV[np.argmin(resultsV[:,6])], resultsT[np.argmin(resultsV[:,6])]])

X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state = 0)
#RegressionWiSARDExperiment(X_train, y_train, X_test, y_test)
#NTupleRegressorExperiment(X_train, y_train, X_test, y_test)
#DecitionTreeExperiment(X_train, y_train, X_test, y_test)
#RandomForestExperiment(X_train, y_train, X_test, y_test)
#XGBoostExperiment(X_train, y_train, X_test, y_test)
ClusRegressionWiSARDExperiment(X_train, y_train, X_test, y_test)
