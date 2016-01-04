import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn import cluster
from sklearn import cross_validation
import patsy
from collections import Counter
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import multiprocessing

CPU_CNT = multiprocessing.cpu_count()

types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}

trainDF = pd.read_csv("data/train.csv", parse_dates=[2], dtype=types)
trainDF[["DayOfWeek"]] = trainDF[["DayOfWeek"]].astype(str)
trainDF[["StateHoliday"]] = trainDF[["StateHoliday"]].astype(str)

testDF = pd.read_csv("data/test.csv", parse_dates=[3], dtype=types)
testDF[["DayOfWeek"]] = testDF[["DayOfWeek"]].astype(str)
testDF[["StateHoliday"]] = testDF[["StateHoliday"]].astype(str)

storeDF = pd.read_csv("data/store.csv")
trainDF = trainDF.merge(storeDF, on=["Store"])
testDF = testDF.merge(storeDF, on=["Store"])

storeList = list(set(trainDF["Store"].values))

clusterDF = trainDF.copy()
clusterDF["Month"] = clusterDF["Date"].dt.month

groups = clusterDF.set_index(["Store", "Month"]).groupby(level=[0, 1])["Sales"]
means = groups.mean()
stds = groups.std(ddof=0)

clusterDF = means.reset_index()
clusterDF["STD"] = stds.values
clusterDF.columns = ["Store", "Month", "MEAN", "STD"]
clusterDF = clusterDF.set_index(["Store", "Month"]).unstack()
clusterDF.columns = ['_'.join(str(x)) for x in clusterDF.columns]
clusterDF.columns = [
    x.replace("_", "").replace("(", "").replace(")", "").replace(",", "").replace(" ", "").replace("'", "") for x in
    clusterDF.columns]

clusterDF = np.log(clusterDF + 1)
clusterDF = clusterDF.reset_index().merge(
    storeDF[["Store", "StoreType", "Assortment"]],
    on=["Store"]
).set_index("Store")
storeList = clusterDF.index

clusterDFMatrix = patsy.dmatrix("%s-1" % ('+'.join(list(clusterDF.columns))), clusterDF)
clusterList = cluster.KMeans(n_clusters=100).fit_predict(clusterDFMatrix)
print Counter(clusterList)


def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)


def getRegressor(_train, _target, _test):
    _train["Sales"] = _target
    _train.columns = map(lambda x: x.replace("[", "").replace("]", ""), _train.columns)

    X_train, X_valid = cross_validation.train_test_split(_train, test_size=0.10)

    y_train = X_train["Sales"]
    y_valid = X_valid["Sales"]

    X_train = X_train.drop("Sales", 1)
    X_valid = X_valid.drop("Sales", 1)

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    params = {
        "objective": "reg:linear",
        "booster": "gbtree",
        "eta": 0.02,  # 0.25 0.06, #0.01,
        "max_depth": 10,
        "subsample": 0.9,  # 0.7
        "colsample_bytree": 0.7,  # 0.7
        "silent": 1
    }
    num_boost_round = 3000
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
                    early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=False, maximize=False)

    _test.columns = map(lambda x: x.replace("[", "").replace("]", ""), _test.columns)
    dtest = xgb.DMatrix(_test)
    return gbm.predict(dtest)


def build_features(features, data):
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])

    features.extend(['StoreType', 'Assortment', 'StateHoliday'])

    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
                              (data.Month - data.CompetitionOpenSinceMonth)

    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
                        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    features.append('IsPromoMonth')
    month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    data['Year'] = data['Year'].astype(str)
    data['Month'] = data['Month'].astype(str)

    return data


results = {}

for clusterIndex, cnt in Counter(clusterList).iteritems():
    print clusterIndex, cnt
    l = storeList[np.where(clusterList == clusterIndex)]

    train = trainDF[(trainDF["Sales"] > 0)].copy()
    test = testDF.copy()

    train = train[train["Store"].isin(l)]
    test = test[test["Store"].isin(l)]

    target = np.log1p(train["Sales"].values)
    testIds = test["Id"].values

    features = []
    build_features(features, train)
    build_features([], test)
    df = pd.concat([train, test], ignore_index=True)

    df = df[features]
    df["Store"] = df["Store"].astype(str)
    dfMatrix = patsy.dmatrix("%s-1" % ('+'.join(list(df.columns))), df, return_type="dataframe")

    predictedY = getRegressor(
        dfMatrix.head(len(train)),
        target,
        dfMatrix.tail(len(test))
    )

    for i in range(len(testIds)):
        results[testIds[i]] = np.expm1(predictedY[i])

submissionDF = pd.read_csv("data/sample_submission.csv")
for k, v in results.iteritems():
    submissionDF.loc[submissionDF["Id"] == k, "Sales"] = int(v)
submissionDF.to_csv("output/submission.csv", index=False)
