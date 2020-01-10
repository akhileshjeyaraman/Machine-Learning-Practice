# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:38:38 2020

@author: Akhilesh 
"""

import pandas as pd
import numpy as np
import math, quandl
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style


df = quandl.get('WIKI/GOOGL')

#Selecting relevant columns from the dataset
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_perct'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0 
df['Perct_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0 

df = df[['Adj. Close', 'HL_perct', 'Perct_change', 'Adj. Volume']]

df['forecast_col'] = df['Adj. Close']
df.fillna(-9999, inplace = True)      #-9999 so that it can be treated as an outlier by the classifier

forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df['forecast_col'].shift(-forecast_out)
df.dropna(inplace = True)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
df.dropna(inplace=True)
y = np.array(df['label'])
X_lately = X[-forecast_out:]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan
print(forecast_set)

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
