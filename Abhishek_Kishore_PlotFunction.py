# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:29:28 2019

@author: gkabh
"""
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from math import floor
from sklearn import linear_model
from datetime import datetime, timedelta
import statsmodels.regression.linear_model as rg



#Function to get the historical 10 years data for the given ticker
def get_data(ticker):
    
    ticker = pd.DataFrame(pdr.get_data_yahoo(ticker, 
                          start=datetime(2012, 1, 1), 
                          end=datetime(2019, 1, 1)))
    return ticker

#Function to calculate the spread between 2 stocks
def reg(x,y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x,pd.Series([1]*len(x),index = x.index)], axis=1)
    regr.fit(x_constant, y)    
    beta = regr.coef_[0]
    alpha = regr.intercept_
    spread = y - x*beta - alpha
    return spread


#Function to plot the data
def plot_function(X,Y1,Y2,X_label,Y1_label,Y2_label):
    trace1 = go.Scatter(
    x=X,
    y=Y1,
    name='XOM'
    )
    trace2 = go.Scatter(
    x=X,
    y=Y2,
    name='CVX',
    yaxis='y2'
    )
    data = [trace1, trace2]
    layout = go.Layout(
    title= Y1_label +" " + "VS" + " " + Y2_label,
    xaxis=dict(
        title=X_label
    ),
    yaxis=dict(
        title=Y1_label
    ),
    yaxis2=dict(
        title=Y2_label,
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
    )
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='multiple-axes-double')
  
    #Function to plot the spread of the 2 Stocks calculated using reg function defined above
def plot_spread(X,Y):
    trace0 = go.Scatter(
    x=X,
    y = Y
    )
    data = [trace0]
    layout = {
    'title' : "Pairs Spread",
    'shapes': [
        # Line Horizontal
        {
            'type': 'line',
            'x0': min(X),
            'y0': Y.mean(),
            'x1': max(X),
            'y1': Y.mean(),
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 4,
                'dash': 'dashdot',
            },
        },
        
    ]
    }

    fig1 = {
    'data': data,
    'layout': layout,
    }

    plot(fig1, filename='shapes-lines')

#Function to plot the rolling Spread Z Score with thresholds     
def plot_spreadmeans(X,Y):
    trace0 = go.Scatter(
    x= X,
    y = Y
    )
    data = [trace0]
    layout = {
    'title' : "Rolling Spread Z Score",
    'shapes': [
        # Line Horizontal
        {
            'type': 'line',
            'x0': min(X),
            'y0': -2,
            'x1': max(X),
            'y1': -2,
            'line': {
                'color': 'green',
                'width': 2,
                #'dash': 'dashdot',
            },
        },
        {
            'type': 'line',
            'x0': min(X),
            'y0': -1,
            'x1': max(X),
            'y1': -1,
            'line': {
                'color': 'green',
                'width': 2,
                'dash': 'dashdot',
            },
        },
        {
            'type': 'line',
            'x0': min(X),
            'y0': 2,
            'x1': max(X),
            'y1': 2,
            'line': {
                'color': 'red',
                'width': 2,
                #'dash': 'dashdot',
            },
        },
        {
            'type': 'line',
            'x0': min(X),
            'y0': 1,
            'x1': max(X),
            'y1': 1,
            'line': {
                'color': 'red',
                'width': 2,
                'dash': 'dashdot',
            },
        },
        
    ]
    }

    fig1 = {
    'data': data,
    'layout': layout,
    }

    plot(fig1, filename='shapes-lines')

