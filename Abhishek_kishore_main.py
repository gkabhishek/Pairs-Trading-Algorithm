# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 01:08:24 2019

@author: gkabh
"""

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime
import statsmodels.regression.linear_model as rg
from plotfunction import*
import arch.unitroot as at

#Get historical data for required stock
stock1 = get_data("XOM")
stock2 = get_data("CVX")

data = pd.concat([stock1['Adj Close'],stock2['Adj Close']], axis = 1, sort = False)
data.columns = ['stock1_AdjClose','stock2_AdjClose']

#Splitting into Training and Test data
train = data[:'2016-12-31']
test = data['2017-01-01':]

#Calculating returns and returns correlation to check for inital correlation between the pairs
#in the training Dataset
returns_stock1 = pd.DataFrame(train['stock1_AdjClose'].pct_change(1).dropna())
returns_stock1.columns = ['Daily_Returns']
returns_stock2 = pd.DataFrame(train['stock2_AdjClose'].pct_change(1).dropna())
returns_stock2.columns = ['Daily_Returns']

print(returns_stock1.join(returns_stock2, on=returns_stock1.index, how='left',lsuffix='_Stock1', rsuffix='_Stock2').corr())
#Shows that returns of both the stocks are highly correlated since 0.81 ic correlation coefficient

#Plotting the daily Prices for Both the Stocks
plot_function(train.index,train['stock1_AdjClose'],train['stock2_AdjClose'],'Years',"XOM","CVX")

# Pairs Spread Cointergeration Calculation and test if that spread is stationary
# Spread is defined as assets prices linear regression residuals or forecasting errors
#Spread Calculation
train_spread = pd.DataFrame(train['stock1_AdjClose'] - rg.OLS(train['stock1_AdjClose'],train['stock2_AdjClose']).fit().params[0] * train['stock2_AdjClose'])
train_spread.columns = ['Train_Spread']

plot_spread(train_spread.index,train_spread['Train_Spread'])

#Engel Granger Test using ADF and Phillip Perron test
#Step1: Induvidual Stock prices should not be Stationary

print('== XOM Prices Augmented Dickey-Fuller Test ==')
print('')
print(at.ADF(train['stock1_AdjClose'], trend='ct'))
print('')
print('== CVX Prices Augmented Dickey-Fuller Test ==')
print('')
print(at.ADF(train['stock2_AdjClose'], trend='ct'))
print('')

#Step2: Check is stock prices become Stationary when differenced once

print('== XOM Prices Differences Augmented Dickey-Fuller Test ==')
print('')
print(at.ADF(train['stock2_AdjClose'].diff(1).dropna(), trend='ct'))
print('')
print('== CVX Prices Differences Augmented Dickey-Fuller Test ==')
print('')
print(at.ADF(train['stock2_AdjClose'].diff(1).dropna(), trend='ct'))
print('')

#Step3: Check if Spread between the assests is Stationary(Cointegeration tests)

print('== XOM-CVX Spread Augmented Dickey-Fuller Co-Integration Test ==')
print('')
print(at.ADF(train_spread['Train_Spread'], trend='ct'))
print('')
print('==  XOM-CVX Spread Phillips-Perron Co-Integration Test ==')
print('')
print(at.PhillipsPerron(train_spread['Train_Spread'], trend='ct', test_type='rho'))
print('')

#Pairs Trading Strategy
#Rolling Spread Z Score

test_spread = pd.DataFrame(test['stock1_AdjClose'] - rg.OLS(test['stock1_AdjClose'],test['stock2_AdjClose']).fit().params[0] * test['stock2_AdjClose'])
Rolling_spread_Z = (test_spread - test_spread.rolling(window = 21).mean())/test_spread.rolling(window = 21).std()
Rolling_spread_Z.columns = ['Rolling_Z']

#Visualizing the Rolling Spread Z Score and its bands
plot_spreadmeans(Rolling_spread_Z.index,Rolling_spread_Z['Rolling_Z'])

#Calculating Trading Signals
#Add rolling Spread Z score to test data and calculate previous period's and Second Previous
#Rolling Spread Z score

test.insert(len(test.columns),'Rolling_spread_Z',Rolling_spread_Z) 

#Calculating previous and second previous rolling spread Z Score
test.insert(len(test.columns),'Rolling_spread_Z(-1)',Rolling_spread_Z.shift(1)) 
test.insert(len(test.columns),'Rolling_spread_Z(-2)',Rolling_spread_Z.shift(2)) 

#Generating trading sigals and appending it to test data
trading_signal = 0
trading_signal_array = []

for i,row in test.iterrows():
    if (row['Rolling_spread_Z(-2)']) > -2 and row['Rolling_spread_Z(-1)'] < -2:
        trading_signal = -2.0
    elif (row['Rolling_spread_Z(-2)']) < -1 and row['Rolling_spread_Z(-1)'] > -1:
        trading_signal = -1.0
    elif (row['Rolling_spread_Z(-2)']) < 2 and row['Rolling_spread_Z(-1)'] > 2:
        trading_signal = 2.0
    elif (row['Rolling_spread_Z(-2)']) > 1 and row['Rolling_spread_Z(-1)'] < 1:
        trading_signal = 1.0
    else:
        trading_signal = 0.0
    trading_signal_array.append(trading_signal)
    
test.insert(len(test.columns), 'trading_signal', trading_signal_array)

#Generate Trading Position using Trading Signals

trading_position = 0.0
trading_position_array = []

for i,row in test.iterrows():
    if (row['trading_signal']) == -2.0:
        trading_position = 1.0
    elif (row['trading_signal']) == -1.0:
        trading_position = 0.0
    elif (row['trading_signal']) == 2.0:
        trading_position = -1.0
    elif (row['trading_signal']) == 1.0:
        trading_position = 0.0
    else:
        trading_position = None
    trading_position_array.append(trading_position)
test.insert(len(test.columns), 'trading_position', trading_position_array)

test = test.fillna(method='ffill')

print(test[['Rolling_spread_Z','trading_signal','trading_position']].tail())

#Performance Evaluation
#Calculate daily returns and Spread Returns
#first two are buy and hold benchmarks
returns_stock1_test = test['stock1_AdjClose'].pct_change(1).dropna()
returns_stock2_test = test['stock2_AdjClose'].pct_change(1).dropna()
#Spread Returns
returns_test = returns_stock1_test -rg.OLS(train['stock1_AdjClose'],train['stock2_AdjClose']).fit().params[0] * returns_stock2_test 

#Strategy Returns Without Trading Commissions Returns
returns_test_no_commison = returns_test * test['trading_position']

#Strategy With Trading Commissions Returns (0.01% Per Trade)
test.insert(len(test.columns), 'trading_position(-1)', test['trading_position'].shift(1))

trading_commision = 0.0
trading_commision_array = []

for i,row in test.iterrows():
    if (row['trading_signal'] == -2.0 or row['trading_signal'] == -1.0 or row['trading_signal'] == 2.0
        or row['trading_signal'] == 1.0) and row['trading_position'] != row['trading_position(-1)']:
        trading_commision = 0.001
    else:
        trading_commision = 0.00
    trading_commision_array.append(trading_commision)
test.insert(len(test.columns), 'trading_commision', trading_commision_array)

returns_test_cummulative = returns_test_no_commison - test['trading_commision']

#Calculating cummulative Annualized Returns
#Strategy without commision
Ann_ret_no_commision = np.cumprod(returns_test_no_commison + 1)**(252/len(test)) - 1
Ann_ret_commision = np.cumprod(returns_test_cummulative + 1)**(252/len(test)) - 1
Ann_ret_stock1 = np.cumprod(returns_stock1_test + 1)**(252/len(test)) - 1
Ann_ret_stock2 = np.cumprod(returns_stock2_test + 1)**(252/len(test)) - 1

# Strategy Performance Summary
results4 = [{'0': 'Annualized:', '1': 'returns_test_without_commison', '2': 'returns_test_with_commision', '3': 'returns_stock1_test', '4': 'returns_stock2_test'},
        {'0': 'Return', '1': np.round(abs(Ann_ret_no_commision[-1]), 4),
         '2':np.round(abs(Ann_ret_commision[-1]), 4),
         '3': np.round(Ann_ret_stock1[-1], 4),
         '4': np.round(Ann_ret_stock2[-1], 4)},
        {'0': 'Standard Deviation', '1': np.round(np.std(returns_test_no_commison) * np.sqrt(252), 4),
         '2': np.round(np.std(returns_test_cummulative) * np.sqrt(252), 4),
         '3': np.round(np.std(returns_stock1_test) * np.sqrt(252), 4),
         '4': np.round(np.std(returns_stock2_test) * np.sqrt(252), 4)},
        {'0': 'Sharpe Ratio (Rf=0%)', '1': np.round(abs(Ann_ret_no_commision[-1] / (np.std(returns_test_no_commison) * np.sqrt(252))), 4),
         '2': np.round(abs(Ann_ret_commision[-1] / (np.std(returns_test_cummulative) * np.sqrt(252))), 4),
         '3': np.round(Ann_ret_stock1[-1] / (np.std(returns_stock1_test) * np.sqrt(252)), 4),
         '4': np.round(Ann_ret_stock2[-1] / (np.std(returns_stock2_test) * np.sqrt(252)), 4)}]
table4 = pd.DataFrame(results4)
print('')
print('== Strategy Performace Summary ==')
print('')
print(table4)