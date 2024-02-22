# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 08:56:04 2022

"""
from sklearn import preprocessing
from DeepRVFL_.DeepRVFL import DeepRVFL
import pandas as pd
import numpy as np
from itertools import product,combinations
from sklearn.linear_model import Ridge
import ForecastLib
import pickle
import matplotlib.pyplot as plt
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
def format_data(dat,target,order,step=1):
    n_sample=dat.shape[0]-order-step+1
    x=np.zeros((n_sample,dat.shape[1]*order))
    y=np.zeros((n_sample,1))
    for i in range(n_sample):
        x[i,:]=dat[i:i+order,:].ravel()
        y[i]  =target[i+order+step-1]
    return x.T,y.T
def format_disc_data(dat,target,order,max_lag=12,step=1):
    n_sample=dat.shape[0]-max_lag
    x=np.zeros((n_sample,len(order)*dat.shape[1]))
    y=np.zeros((n_sample,1))
    for i in range(n_sample):
        x[i,:]=dat[i:i+max_lag,:][order,:].ravel()
        y[i]  =target[i+max_lag+step-1,0]
    return x,y
def pimse_pre(x_,y_,pre_l,hyper):
    order=hyper[0]
    features_comb=hyper[1]
    alpha=hyper[2]
    start=hyper[3]
    x,y=format_disc_data(x_[:,features_comb],y_,order,max_lag=12)
    trainx,trainy=x[:-pre_l:],y[:-pre_l,:]
    testx=x[-pre_l:,:]
    model=Ridge(alpha=alpha)
    model.fit(trainx,trainy.ravel())
    prediction=model.predict(testx)
    allp=model.predict(x)
    return prediction,allp
def pimse_fs(x_,y_,pre_l):
    xscaler=preprocessing.MinMaxScaler()
    xscaler.fit(x_[:-pre_l,:])
    x_=xscaler.transform(x_)    
    yscaler=preprocessing.MinMaxScaler()
    yscaler.fit(y_[:-pre_l,:])
    y_=yscaler.transform(y_)
    feats_comb=[]
    for i in range(1,len(feats)):
        a=list(combinations(list(range(len(feats))),i))
        for j in range(len(a)):
            feats_comb.append(a[j])
    orders=[]
    max_lag=12

    lags=[0,1,10,11]
    for i in range(1,4):        
        a=list(combinations(list(range(max_lag)),i))
        for j in a:
            orders.append(j)
    alphas=[0]
    starts=[0]
    hypers=list(product(orders,feats_comb,alphas,starts))
  
    loss=[]
    testy=y_[-pre_l:]
    for i, h in enumerate(hypers[:]):
        p,_=pimse_pre(x_,y_,pre_l,h)
        e=compute_error(testy, p)
        loss.append(e['MAE'])
    minl=min(loss)
    bh=hypers[loss.index(minl)]
    id1 = [i for i,x in enumerate(loss) if x==minl]
    bhs=[hypers[i]for i in id1]
    return bh,bhs
def round_(x):
    l=len(x)
    r=[]
    for i in range(l):
        ele=x[i]
        int_=int(ele)
        if int_-0.25<ele<=int_+0.25:
            r.append(int_)
        elif int_+0.25<ele<=int_+0.75:
            r.append(int_+0.5)
        else:
            r.append(int_+1)
    return np.array(r)
            
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
    print(targf)            
    testl=36
    vall=36
    target=data[targf].values.astype(np.float64)
    dat=data[feats].values.astype(np.float64)
    
    x_=dat[:-testl,:]
    y_=target[:-testl].reshape(-1,1)
    if dif1st:
        dif_dat=dat[1:-testl,:]-dat[:-testl-1,:]
        dif_target=target[1:-testl].reshape(-1,1)-target[:-testl-1].reshape(-1,1)
    
    pimsebh,bhs=pimse_fs(x_[:-testl,:],y_[:-testl],vall)
    print(pimsebh)
    if '/' in targf:
        targf=targf.replace('/',' ')
    name=targf+'Hyperstep'+str(step)+'.pkl'
    f=open(name,'wb')
    pickle.dump(pimsebh,f)
    f.close()
    
    xscaler=preprocessing.MinMaxScaler()
    xscaler.fit(dat[:-testl,:])
    x_=xscaler.transform(dat)
    
    yscaler=preprocessing.MinMaxScaler()
    yscaler.fit(target[:-testl].reshape(-1,1))
    y_=yscaler.transform(target.reshape(-1,1))

    predictions = []
    for seed in np.arange(10):
        np.random.seed(seed)

        
        model = pimse_pre(x_,y_,testl,pimsebh)
        ppre,allp=pimse_pre(x_,y_,testl,pimsebh)
        allpredictions=yscaler.inverse_transform(allp.reshape(-1,1))
        pimsepredictions=yscaler.inverse_transform(ppre.reshape(-1,1))
        predictions.append(pimsepredictions)
    pimsepre=np.concatenate(predictions,axis=1)
    pimsepre=pd.DataFrame(pimsepre)
    pimsepre.to_csv(targf+'Step'+str(step)+'PIMSE.csv')
   