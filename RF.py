# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 08:56:04 2022

@author: ruobi
"""
from sklearn.svm import SVR
from utils import MSE, config_MG, load_MG
from sklearn import preprocessing
from DeepRVFL_.DeepRVFL import DeepRVFL
import pandas as pd
import numpy as np
from itertools import product,combinations
from sklearn.linear_model import Ridge
import ForecastLib
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from nbeats import TsMetric, setup_seed
def compute_error(actuals,predictions,history=None):
    actuals=actuals.ravel()
    predictions=predictions.ravel()    
    metric=ForecastLib.TsMetric()
    error={}
    error['RMSE']=metric.RMSE(actuals, predictions)
    error['MAE']=metric.MAE(actuals,predictions)
    if history is not None:
        history=history.ravel()
        error['MASE']=metric.MASE(actuals,predictions,history)    
    return error
def format_data(x_,y_,order):
    n_sample=x_.shape[0]-order
    x=np.zeros((n_sample,order*x_.shape[1]))
    y=np.zeros((n_sample,y_.shape[1]))
    for i in range(n_sample):
        x[i,:]=x_[i:i+order,:].ravel()
        y[i,:]  =y_[i+order,:]
    return x,y
def RF_pre(trainx,trainy,testx,hyper):
    order=hyper[0]
    n_estimators=hyper[1]
    max_depth=hyper[2]
    min_samples_split=hyper[3]
    model=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split)
    model.fit(trainx, trainy.ravel())
    output=model.predict(testx)
    return output
def cv(x_,y_,pre_l):
    orders=[6,8,10,12]
    n_estimators=[20,40,80]
    max_depths=[2,4,8]
    min_samples_splits=[2,4,8]
    losses=[]
    bhs=[]
    scaler=preprocessing.MinMaxScaler()
    scaler.fit(x_[:-pre_l,:])
    x_=scaler.transform(x_)
    for order in orders:
        allx,ally=format_data(x_,y_,order)
        trainx,trainy=allx[:-pre_l,:],ally[:-pre_l,:]
        testx,testy=allx[-pre_l:,:],ally[-pre_l:,:]
        hypers=list(product([order],n_estimators,max_depths,min_samples_splits))
        loss=[]
        for i, h in enumerate(hypers):
            p=RF_pre(trainx,trainy,testx,h)
            e=compute_error(testy, p)
            loss.append(e['RMSE'])
        minl=min(loss)
        bhs.append(hypers[loss.index(minl)])
        losses.append(minl)
    bh=bhs[losses.index(min(losses))]
    return bh
data=pd.read_csv('Data.csv',index_col=0)
start='Jan-96'
end='Dec-20'
data=data.loc[start:end].astype(float)
variables=data.columns
suezmax=[]
panamax=[]
aframax=[]
capesize=[]
for v in variables:    
    if 'Suezmax' in v:
        suezmax.append(v)
    if 'Panamax' in v:
        panamax.append(v)
    if 'Capesize' in v:
        capesize.append(v)
    if 'Aframax' in v:
        aframax.append(v)
aframax=[
    '113-115K DWT Aframax Tanker Newbuilding Prices',
 'Aframax D/H 105K DWT 5 Year Old Secondhand Prices',
 'Aframax Tanker Orderbook',
   '1 Year Timecharter Rate Aframax (Long Run Historical Series)',

 ]
suezmax=[
    '156-158K DWT Suezmax Tanker Newbuilding Prices',
    'Suezmax D/H 160K DWT 5 Year Old Secondhand Prices',
 'Suezmax Orderbook',
   '1 Year Timecharter Rate Suezmax (Long Run Historical Series)', 
 ]
panamax=[
    '75-77K DWT Panamax Bulkcarrier Newbuilding Prices',
 'Panamax 76K Bulkcarrier 5 Year Old Secondhand Prices',
 'Panamax Bulker Orderbook',
   '1 Year Timecharter Rate Panamax Bulkcarrier (Long Run Historical Series)',
 ]
capesize=[
    '176-180K DWT Capesize Bulkcarrier Newbuilding Prices',
 'Capesize 5 Year Old Secondhand Prices (Long Run Historical Series)',
 'Capesize Bulker Orderbook',
   '1 Year Timecharter Rate Capesize Bulkcarrier (Long Run Historical Series)',
]

dif1st=False
step=1
for feats in [aframax,suezmax,panamax,capesize][:]:    
    targf=feats[2]
    testl=36
    vall=36
    target=data[targf].values.astype(np.float64)   
    dat=data[feats].values.astype(np.float64)
    
    x_=dat[:-testl,:]
    y_=target[:-testl].reshape(-1,1)
    if dif1st:
        dif_dat=dat[1:-testl,:]-dat[:-testl-1,:]
        dif_target=target[1:-testl].reshape(-1,1)-target[:-testl-1].reshape(-1,1)
    
    if '/' in targf:
        targf=targf.replace('/',' ')
    
    bh=cv(x_[:-testl,:],y_[:-testl],vall)
    
    xscaler=preprocessing.MinMaxScaler()
    xscaler.fit(dat[:-testl,:])
    x_=xscaler.transform(dat)
    
    yscaler=preprocessing.MinMaxScaler()
    yscaler.fit(target[:-testl].reshape(-1,1))
    y_=yscaler.transform(target.reshape(-1,1))
    
    
    x,y=format_data(x_, y_, bh[0])
    trainx,trainy=x[:-testl,:],y[:-testl]
    testx=x[-testl:,:]
    predictions = []

    for seed in np.arange(10):
        np.random.seed(seed)
         
        model = RF_pre(trainx, trainy, testx, bh)
        rfpre=RF_pre(trainx,trainy,testx,bh)
        rfpre=yscaler.inverse_transform(rfpre.reshape(-1,1))
        predictions.append(rfpre)
        
    rfpre=np.concatenate(predictions,axis=1)
    rfpre=pd.DataFrame(rfpre)
    rfpre.to_csv(targf+'Step'+str(step)+'RF.csv')
