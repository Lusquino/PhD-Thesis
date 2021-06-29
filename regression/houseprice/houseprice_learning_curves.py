#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import matplotlib.style as style
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import math

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("HousePrice/train.csv")
train.head()

test = pd.read_csv("HousePrice/test.csv")
test.head()

def missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_percentage(train)

missing_percentage(test)

def plotting_3_chart(df, feature):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('QQ_plot')
    stats.probplot(df.loc[:,feature], plot = ax2)
    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title('Box Plot')
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

plotting_3_chart(train, 'SalePrice')

(train.corr()**2)["SalePrice"].sort_values(ascending = False)[1:]

train = train[train.GrLivArea < 4500]
train.reset_index(drop = True, inplace = True)
previous_train = train.copy()

plt.subplots(figsize = (15,10))
sns.residplot(train.GrLivArea, train.SalePrice);

plotting_3_chart(train, 'SalePrice')

train["SalePrice"] = np.log1p(train["SalePrice"])

plotting_3_chart(train, 'SalePrice')

fig, (ax1, ax2) = plt.subplots(figsize = (20,6), ncols=2, sharey = False, sharex=False)
sns.residplot(x = previous_train.GrLivArea, y = previous_train.SalePrice, ax = ax1)
sns.residplot(x = train.GrLivArea, y = train.SalePrice, ax = ax2);

style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,20))

mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(train.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0, );
plt.title("Heatmap of all the Features", fontsize = 30);

train.drop(columns=['Id'],axis=1, inplace=True)
test.drop(columns=['Id'],axis=1, inplace=True)

y = train['SalePrice'].reset_index(drop=True)

previous_train = train.copy()

all_data = pd.concat((train, test)).reset_index(drop = True)
all_data.drop(['SalePrice'], axis = 1, inplace = True)

missing_percentage(all_data)

missing_val_col = ["Alley",
                   "PoolQC",
                   "MiscFeature",
                   "Fence",
                   "FireplaceQu",
                   "GarageType",
                   "GarageFinish",
                   "GarageQual",
                   "GarageCond",
                   'BsmtQual',
                   'BsmtCond',
                   'BsmtExposure',
                   'BsmtFinType1',
                   'BsmtFinType2',
                   'MasVnrType']

for i in missing_val_col:
    all_data[i] = all_data[i].fillna('None')

missing_val_col2 = ['BsmtFinSF1',
                    'BsmtFinSF2',
                    'BsmtUnfSF',
                    'TotalBsmtSF',
                    'BsmtFullBath',
                    'BsmtHalfBath',
                    'GarageYrBlt',
                    'GarageArea',
                    'GarageCars',
                    'MasVnrArea']

for i in missing_val_col2:
    all_data[i] = all_data[i].fillna(0)

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Utilities'] = all_data['Utilities'].fillna('AllPub')
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA")
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")

missing_percentage(all_data)

sns.distplot(all_data['1stFlrSF']);

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_feats

def fixing_skewness(df):
    from scipy.stats import skew
    from scipy.special import boxcox1p
    from scipy.stats import boxcox_normmax

    numeric_feats = df.dtypes[df.dtypes != "object"].index

    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    skewed_features = high_skew.index

    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))

fixing_skewness(all_data)

sns.distplot(all_data['1stFlrSF']);

all_data = all_data.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']

all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +
                                 all_data['1stFlrSF'] + all_data['2ndFlrSF'])

all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
                              all_data['WoodDeckSF'])

all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

all_data.shape

final_features = pd.get_dummies(all_data).reset_index(drop=True)
final_features.shape

X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(y):, :]

outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])

def overfit_reducer(df):
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99.94:
            overfit.append(i)
    overfit = list(overfit)
    return overfit

overfitted_features = overfit_reducer(X)

X = X.drop(overfitted_features, axis=1)
X_sub = X_sub.drop(overfitted_features, axis=1)

X.shape,y.shape, X_sub.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state = 0)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

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
    possibleAddr = {}
    means = [wsd.PowerMean(2), wsd.HarmonicMean(), wsd.HarmonicPowerMean(2), wsd.GeometricMean(), wsd.ExponentialMean(), wsd.SimpleMean(),wsd.Median()]

    mins = [ np.min(X_train[:,i]) for i in range(X_train.shape[1]) ]
    maxs = [ np.max(X_train[:,i]) for i in range(X_train.shape[1]) ]

    for t in range(15,16):
        therm = [t for i in range(X_train.shape[1])]
        sTherm = np.sum(therm)

        if sTherm not in possibleAddr.keys():
            possibleAddr[sTherm] = allDividers(sTherm)

        DT = wsd.DynamicThermometer(therm, mins, maxs)

        binX_train = [DT.transform(X_train[i]) for i in range(X_train.shape[0])]
        binX_test = [DT.transform(X_test[i]) for i in range(X_test.shape[0])]

        DS_base = wsd.DataSet(binX_train,y_train)
        DS_test = wsd.DataSet(binX_test,y_test)

        for addr in possibleAddr[sTherm]:
            for m in means:
                maets = []
                msets = []

                rwsd = wsd.RegressionWisard(addressSize=addr, mean=m, completeAddressing=True)
                for i in range(1, 2):
                    print(i,m,addr,t)
                    for dt_size in range(0, len(DS_base)):
                        DS_train = wsd.DataSet()
                        for j in range(0, dt_size+1):
                            DS_train.add(DS_base.get(j), DS_base.getY(j))
                        rwsd.train(DS_train)
                        outTest = np.array(rwsd.predict(DS_test))
                        outTest[np.isnan(outTest)] = np.nanmean(outTest)
                        outTest[outTest==np.inf] = 0
   
                        if(np.any(np.isnan(outTest))):
                          for j in range(0, len(outTest)):
                              if(math.isnan(outTest[j])):
                                outTest[j] = 0

                        maets.append(mean_absolute_error(outTest,y_test))
                        msets.append(mean_squared_error(outTest,y_test))
    
                        resultsP.append([addr, m])
                        resultsV.append([np.mean(maets), np.std(maets), np.mean(msets), np.std(msets)])

                        saveToFile('housepriceRew_learning_curve.csv', [dt_size, addr, m, np.mean(maets), np.std(maets), np.mean(msets)])

    resultsP = np.array(resultsP)
    resultsV = np.array(resultsV)

    saveToFile('housepriceRew_learning_curve.csv',[resultsP[np.argmin(resultsV[:,2])], resultsV[np.argmin(resultsV[:,2])], resultsT[np.argmin(resultsV[:,2])]])
    saveToFile('housepriceRew_learning_curve.csv',[resultsP[np.argmin(resultsV[:,6])], resultsV[np.argmin(resultsV[:,6])], resultsT[np.argmin(resultsV[:,6])]])

def NTupleExperiment(X_train, y_train, X_test, y_test):
    resultsP = []
    resultsV = []
    resultsT = []

    #preprocessing
    possibleAddr = {}
    means = [wsd.SimpleMean()]

    mins = [ np.min(X_train[:,i]) for i in range(X_train.shape[1]) ]
    maxs = [ np.max(X_train[:,i]) for i in range(X_train.shape[1]) ]

    for t in range(15,16):
        therm = [t for i in range(X_train.shape[1])]
        sTherm = np.sum(therm)

        if sTherm not in possibleAddr.keys():
            possibleAddr[sTherm] = allDividers(sTherm)

        DT = wsd.DynamicThermometer(therm, mins, maxs)

        binX_train = [DT.transform(X_train[i]) for i in range(X_train.shape[0])]
        binX_test = [DT.transform(X_test[i]) for i in range(X_test.shape[0])]

        DS_base = wsd.DataSet(binX_train,y_train)
        DS_test = wsd.DataSet(binX_test,y_test)

        for addr in possibleAddr[sTherm]:
            for m in means:
                maets = []
                msets = []

                rwsd = wsd.RegressionWisard(addressSize=addr, mean=m, completeAddressing=True, orderedMapping = True)
                for i in range(1, 2):
                    print(i,m,addr,t)
                    for dt_size in range(0, len(DS_base)):
                        DS_train = wsd.DataSet()
                        for j in range(0, dt_size+1):
                            DS_train.add(DS_base.get(j), DS_base.getY(j))
                        rwsd.train(DS_train)                          
                        outTest = np.array(rwsd.predict(DS_test))
                        outTest[np.isnan(outTest)] = np.nanmean(outTest)
                        outTest[outTest==np.inf] = 0
    
                        if(np.any(np.isnan(outTest))):
                          for j in range(0, len(outTest)):
                              if(math.isnan(outTest[j])):
                                outTest[j] = 0

                        maets.append(mean_absolute_error(outTest,y_test))
                        msets.append(mean_squared_error(outTest,y_test))
    
                        resultsP.append([addr, m])
                        resultsV.append([np.mean(maets), np.std(maets), np.mean(msets), np.std(msets)])

                        saveToFile('housepriceNTuple_learning_curve.csv', [dt_size, addr, m, np.mean(maets), np.std(maets), np.mean(msets)])

    resultsP = np.array(resultsP)
    resultsV = np.array(resultsV)

    saveToFile('housepriceNTuple_learning_curve.csv',[resultsP[np.argmin(resultsV[:,2])], resultsV[np.argmin(resultsV[:,2])], resultsT[np.argmin(resultsV[:,2])]])
    saveToFile('housepriceNTuple_learning_curve.csv',[resultsP[np.argmin(resultsV[:,6])], resultsV[np.argmin(resultsV[:,6])], resultsT[np.argmin(resultsV[:,6])]])

def ClusRegressionWiSARDExperiment(X_train, y_train, X_test, y_test):
    resultsP = []
    resultsV = []
    resultsT = []

    #preprocessing
    possibleAddr = {}
    means = [wsd.PowerMean(2), wsd.HarmonicMean(), wsd.HarmonicPowerMean(2), wsd.GeometricMean(), wsd.ExponentialMean(), wsd.SimpleMean(),wsd.Median()]
    #means = [wsd.Median()]

    mins = [ np.min(X_train[:,i]) for i in range(X_train.shape[1]) ]
    maxs = [ np.max(X_train[:,i]) for i in range(X_train.shape[1]) ]

    for t in range(15,16):
        therm = [t for i in range(X_train.shape[1])]
        sTherm = np.sum(therm)

        if sTherm not in possibleAddr.keys():
            possibleAddr[sTherm] = allDividers(sTherm)

        DT = wsd.DynamicThermometer(therm, mins, maxs)

        binX_train = [DT.transform(X_train[i]) for i in range(X_train.shape[0])]
        binX_test = [DT.transform(X_test[i]) for i in range(X_test.shape[0])]

        DS_base = wsd.DataSet(binX_train,y_train)
        DS_test = wsd.DataSet(binX_test,y_test)

        for addr in possibleAddr[sTherm]:
            for m in means:
                maets = []
                msets = []

                rwsd = wsd.RegressionWisard(addressSize=addr, mean=m, completeAddressing=True, minScore = 0.1, threshold = 1000)
                for i in range(1, 2):
                    print(i,m,addr,t)
                    for dt_size in range(0, len(DS_base)):
                        DS_train = wsd.DataSet()
                        for j in range(0, dt_size+1):
                            DS_train.add(DS_base.get(j), DS_base.getY(j))
                        rwsd.train(DS_train)
                        #print(len(DS_train))                          
                        outTest = np.array(rwsd.predict(DS_test))
    
                        #print(outTest)
                        outTest[np.isnan(outTest)] = np.nanmean(outTest)
                        outTest[outTest==np.inf] = 0

                        if(np.any(np.isnan(outTest))):
                          for j in range(0, len(outTest)):
                              if(math.isnan(outTest[j])):
                                outTest[j] = 0

                        #print(outTest)
    
                        maets.append(mean_absolute_error(outTest,y_test))
                        msets.append(mean_squared_error(outTest,y_test))
    
                        resultsP.append([addr, m])
                        resultsV.append([np.mean(maets), np.std(maets), np.mean(msets), np.std(msets)])

                        saveToFile('housepriceCRew_learning_curve.csv', [dt_size, addr, m, np.mean(maets), np.std(maets), np.mean(msets)])

    resultsP = np.array(resultsP)
    resultsV = np.array(resultsV)

    saveToFile('housepriceCRew_learning_curve.csv',[resultsP[np.argmin(resultsV[:,2])], resultsV[np.argmin(resultsV[:,2])], resultsT[np.argmin(resultsV[:,2])]])
    saveToFile('housepriceCRew_learning_curve.csv',[resultsP[np.argmin(resultsV[:,6])], resultsV[np.argmin(resultsV[:,6])], resultsT[np.argmin(resultsV[:,6])]])


#RegressionWiSARDExperiment(X_train, y_train, X_test, y_test)
NTupleExperiment(X_train, y_train, X_test, y_test)
#DecitionTreeExperiment(X_train, y_train, X_test, y_test)
#RandomForestExperiment(X_train, y_train, X_test, y_test)    
#XGBoostExperiment(X_train, y_train, X_test, y_test)
#ClusRegressionWiSARDExperiment(X_train, y_train, X_test, y_test)
