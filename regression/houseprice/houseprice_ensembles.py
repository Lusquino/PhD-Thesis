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
import bagging
import boosting

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

    saveToFile('housepriceDT.csv', [mean_absolute_error(y_train,tree_preds_train), mean_squared_error(y_train,tree_preds_train), mean_absolute_error(y_test,tree_preds_test), mean_squared_error(y_test,tree_preds_test)], delta0, delta1, delta2)

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

    saveToFile('housepriceDT.csv', [mean_absolute_error(y_train,forest_preds_train), mean_squared_error(y_train,forest_preds_train), mean_absolute_error(y_test,forest_preds_test), mean_squared_error(y_test,forest_preds_test)], delta0, delta1, delta2)

def XGBoostExperiment(X_train, y_train, X_test, y_test):

    xgb_model = XGBoostRegressor()
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

    saveToFile('housepriceXGB.csv', [mean_absolute_error(y_train,xgb_preds_train), mean_squared_error(y_train,xgb_preds_train), mean_absolute_error(y_test,xgb_preds_test), mean_squared_error(y_test,xgb_preds_test)], delta0, delta1, delta2)

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

                saveToFile('housepriceCrew.csv', [t, addr, m, ms, threshold, limit, np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets), np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])

    resultsP = np.array(resultsP)
    resultsV = np.array(resultsV)
    resultsT = np.array(resultsT)

    saveToFile('housepriceCrew.csv',[resultsP[np.argmin(resultsV[:,2])], resultsV[np.argmin(resultsV[:,2])], resultsT[np.argmin(resultsV[:,2])]])
    saveToFile('housepriceCrew.csv',[resultsP[np.argmin(resultsV[:,6])], resultsV[np.argmin(resultsV[:,6])], resultsT[np.argmin(resultsV[:,6])]])

def BaggingRegressionWiSARDExperiment(X_train, y_train, X_test, y_test):
    resultsP = []
    resultsV = []
    resultsT = []

    #preprocessing
    #means = ["mean", "median", "harmonic", "geometric"]
    means = ["mean"]

    mins = [ np.min(X_train[:,i]) for i in range(X_train.shape[1]) ]
    maxs = [ np.max(X_train[:,i]) for i in range(X_train.shape[1]) ]

    #for t in range(5, 31):
    for t in range(5, 6):
        therm = [t for i in range(X_train.shape[1])]
        sTherm = np.sum(therm)

        DT = wsd.DynamicThermometer(therm, mins, maxs)

        binX_train = [DT.transform(X_train[i]) for i in range(X_train.shape[0])]
        binX_test = [DT.transform(X_test[i]) for i in range(X_test.shape[0])]

        DS_train = wsd.DataSet(binX_train,y_train)
        DS_test = wsd.DataSet(binX_test,y_test)

        for learners in range (700, 800, 100):
            maetr = []
            maets = []
            msetr = []
            msets = []
            t1 = []
            t2 = []
            t3 = []
            #for i in range(10):
            for i in range(1):
                print(i,learners,t)
                ensemble = bagging.Regression_Bagging(DS_train, learners)
                a = time.perf_counter()
                ensemble.ensemble()
                b = time.perf_counter()
                delta1 = b - a
                for m in means:
                    a = time.perf_counter()
                    outTrain = np.array(ensemble.predict(DS_train, mean = m))
                    b = time.perf_counter()
                    delta2 = b - a
                    a = time.perf_counter()
                    outTest = np.array(ensemble.predict(DS_test, mean = m))
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
    
                    resultsP.append([t, learners, m])
                    resultsV.append([np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets)])
                    resultsT.append([np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])
    
                    saveToFile('housepriceBaggingRew_new_new.csv', [t, learners, m, np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets), np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])

    resultsP = np.array(resultsP)
    resultsV = np.array(resultsV)
    resultsT = np.array(resultsT)

    saveToFile('housepriceBaggingRew_new_new.csv',[resultsP[np.argmin(resultsV[:,2])], resultsV[np.argmin(resultsV[:,2])], resultsT[np.argmin(resultsV[:,2])]])
    saveToFile('housepriceBaggingRew_new_new.csv',[resultsP[np.argmin(resultsV[:,6])], resultsV[np.argmin(resultsV[:,6])], resultsT[np.argmin(resultsV[:,6])]])

def BoostRegressionWiSARDExperiment(X_train, y_train, X_test, y_test):
    resultsP = []
    resultsV = []
    resultsT = []

    #preprocessing
    #means = ["mean", "median", "harmonic", "geometric"]
    means = ["mean"]

    mins = [ np.min(X_train[:,i]) for i in range(X_train.shape[1]) ]
    maxs = [ np.max(X_train[:,i]) for i in range(X_train.shape[1]) ]

    y_boost_train = []
    for i in range(0, int(9*len(y_train)/10)):
        y_boost_train.append(y_train[i])

    #for t in range(5,31):
    for t in range(28, 29):
        therm = [t for i in range(X_train.shape[1])]
        sTherm = np.sum(therm)

        DT = wsd.DynamicThermometer(therm, mins, maxs)

        binX_train = [DT.transform(X_train[i]) for i in range(X_train.shape[0])]
        binX_test = [DT.transform(X_test[i]) for i in range(X_test.shape[0])]

        DS_base = wsd.DataSet(binX_train,y_train)
        DS_test = wsd.DataSet(binX_test,y_test)
        
        DS_train = wsd.DataSet()
        DS_val = wsd.DataSet()
        
        for i in range(0, int(9*len(DS_base)/10)):
            DS_train.add(DS_base.get(i), DS_base.getY(i))
        for i in range(int(9*len(DS_base)/10), len(DS_base)):
            DS_val.add(DS_base.get(i), DS_base.getY(i))

        #for learners in range (100, 1100, 100):
        for learners in range(200, 1100, 100):
            maetr = []
            maets = []
            msetr = []
            msets = []
            t1 = []
            t2 = []
            t3 = []
            #for i in range(10):
            for i in range(1):
                print(i,learners,t)
                boost = boosting.Regression_Boost(DS_train, DS_val, learners)
                a = time.perf_counter()
                boost.ensemble()
                b = time.perf_counter()
                delta1 = b - a
                for m in means:
                    a = time.perf_counter()
                    outTrain = np.array(boost.predict(DS_train, mean = m))
                    b = time.perf_counter()
                    delta2 = b - a
                    a = time.perf_counter()
                    outTest = np.array(boost.predict(DS_test, mean = m))
                    b = time.perf_counter()
                    delta3 = b - a
    
                    outTrain[np.isnan(outTrain)] = np.nanmean(outTrain)
                    outTest[np.isnan(outTest)] = np.nanmean(outTest)
                    outTrain[outTrain==np.inf] = 0
                    outTest[outTest==np.inf] = 0
    
                    maetr.append(mean_absolute_error(outTrain,y_boost_train))
                    maets.append(mean_absolute_error(outTest,y_test))
                    msetr.append(mean_squared_error(outTrain,y_boost_train))
                    msets.append(mean_squared_error(outTest,y_test))
    
                    t1.append(delta1)
                    t2.append(delta2)
                    t3.append(delta3)
    
                    resultsP.append([t, learners, m])
                    resultsV.append([np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets)])
                    resultsT.append([np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])
    
                    saveToFile('housepriceBoostRew_new.csv', [t, learners, m, np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets), np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])

    resultsP = np.array(resultsP)
    resultsV = np.array(resultsV)
    resultsT = np.array(resultsT)

    saveToFile('housepriceBoostRew_new.csv',[resultsP[np.argmin(resultsV[:,2])], resultsV[np.argmin(resultsV[:,2])], resultsT[np.argmin(resultsV[:,2])]])
    saveToFile('housepriceBoostRew_new.csv',[resultsP[np.argmin(resultsV[:,6])], resultsV[np.argmin(resultsV[:,6])], resultsT[np.argmin(resultsV[:,6])]])

def NaiveRegressionWiSARDExperiment(X_train, y_train, X_test, y_test):
    resultsP = []
    resultsV = []
    resultsT = []

    #preprocessing
    means = ["mean", "median", "harmonic", "geometric"]

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

        for learners in range (100, 1100, 100):
            maetr = []
            maets = []
            msetr = []
            msets = []
            t1 = []
            t2 = []
            t3 = []
            for i in range(10):
                print(i,learners,t)
                ensemble = bagging.Regression_Bagging(DS_train, learners, partitions = 1)
                a = time.perf_counter()
                ensemble.ensemble()
                b = time.perf_counter()
                delta1 = b - a
                for m in means:
                    a = time.perf_counter()
                    outTrain = np.array(ensemble.predict(DS_train, mean = m))
                    b = time.perf_counter()
                    delta2 = b - a
                    a = time.perf_counter()
                    outTest = np.array(ensemble.predict(DS_test, mean = m))
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
    
                    resultsP.append([t, learners, m])
                    resultsV.append([np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets)])
                    resultsT.append([np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])
    
                    saveToFile('housepriceNaiveRew.csv', [t, learners, m, np.mean(maetr), np.std(maetr), np.mean(maets), np.std(maets),np.mean(msetr), np.std(msetr), np.mean(msets), np.std(msets), np.mean(t1), np.std(t1), np.mean(t2), np.std(t2), np.mean(t3), np.std(t3)])

    resultsP = np.array(resultsP)
    resultsV = np.array(resultsV)
    resultsT = np.array(resultsT)

    saveToFile('housepriceNaiveRew.csv',[resultsP[np.argmin(resultsV[:,2])], resultsV[np.argmin(resultsV[:,2])], resultsT[np.argmin(resultsV[:,2])]])
    saveToFile('housepriceNaiveRew.csv',[resultsP[np.argmin(resultsV[:,6])], resultsV[np.argmin(resultsV[:,6])], resultsT[np.argmin(resultsV[:,6])]])

#RegressionWiSARDExperiment(X_train, y_train, X_test, y_test)
#NTupleRegressorExperiment(X_train, y_train, X_test, y_test)
#DecitionTreeExperiment(X_train, y_train, X_test, y_test)
#RandomForestExperiment(X_train, y_train, X_test, y_test)    
#XGBoostExperiment(X_train, y_train, X_test, y_test)
#ClusRegressionWiSARDExperiment(X_train, y_train, X_test, y_test)
BaggingRegressionWiSARDExperiment(X_train, y_train, X_test, y_test)
#BoostRegressionWiSARDExperiment(X_train, y_train, X_test, y_test)
#NaiveRegressionWiSARDExperiment(X_train, y_train, X_test, y_test)
