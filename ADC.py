# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:09:41 2023

@author: crkkk
"""

from shippingCNN_basics import shipCNN,TSCNN,setup_seed,shipRCNN,TSRCNN
from sklearn import preprocessing
import matplotlib.pyplot as plt
import ForecastLib
from itertools import product,combinations
import numpy as np
import pandas as pd
import seaborn
from statsmodels.tsa.arima.model import ARIMA
def get_data(name):
    file_name = name+'.csv'
    dat = pd.read_csv(file_name)
    dat = dat.fillna(method='ffill')
    return dat,dat.columns
def format_data(dat,order,step=1,idx=0):
    n_sample=300-order
    print(f"n_sample: {n_sample}, dat.shape[1]: {dat.shape[1]}, order: {order}")
    x=np.zeros((n_sample,dat.shape[1],order))
    y=np.zeros((n_sample,1))
    for i in range(n_sample):
        x[i,:,:]=dat[i:i+order,:].T
        y[i]  =dat[i+order,idx]
    return x,y
def cnn_pre(x,h,l,insample):    
    n_ch=h[0]
    pool=h[1]
    order=h[2]
    batch_size=h[3]
    weight_decay=h[4]
    lr=h[5]
    epochs=h[6]
    step=h[-1]
    print(x.shape)
    allx,ally=format_data(x,order,step)

    trainx,trainy=allx[:-l,:,:],ally[:-l,:]
    testx,testy=allx[-l:,:],ally[-l:,:]
    m=TSCNN(n_ch,pool,order,x.shape[1])
    model=m.fit(trainx,trainy,batch_size=batch_size,weight_decay=weight_decay,lr=lr,epochs=epochs)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data.numpy())
    if insample is True:
        print('cnn',allx.shape)
        cnn_pre_=m.predict(allx,model)
    else:
        cnn_pre_=m.predict(testx,model)
    return cnn_pre_,model
def rcnn_pre(x,h,l,insample,cv=True):    
    n_ch=h[0]
    pool=h[1]
    order=h[2]
    batch_size=h[3]
    weight_decay=h[4]
    lr=h[5]
    epochs=h[6]
    rnn_hidden=h[7]
    step=h[-1]
    allx,ally=format_data(x,order,step)

    trainx,trainy=allx[:-l,:,:],ally[:-l,:]
    testx,testy=allx[-l:,:],ally[-l:,:]
    m=TSRCNN(n_ch,order,rnn_hidden,x.shape[1],lr,epochs)
    model,losses=m.fit(trainx,trainy,batch_size=batch_size,weight_decay=0.0,lr=lr,epochs=epochs)
    fig, ax = plt.subplots(figsize=(10,7))

    ax.grid(color='white')
    ax.plot(losses)
    ax.set_ylabel('Training loss')
    ax.set_xlabel('Epoch')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend(framealpha=0)
    ax.set_facecolor('lightgrey')
    fig.show()

    if insample is True:
        print('cnn',allx.shape)
        cnn_pre_=m.predict(allx,model)
    else:
        cnn_pre_=m.predict(testx,model)
    if cv:
        return cnn_pre_
    else:
        return cnn_pre_,model
def round_(x):
    fn=x-int(x)
    if 0.75>fn >=0.25:
        x=int(x)+0.5
    elif fn>=0.75:
       
        x=int(x)+1
  
    else:
        x=int(x)
    return x
def cv(hypers,x,target_idx,l):
    cnn_losses=[]
    for h in hypers:
        print(cnn_pre(x, h, l, False))
        pre=cnn_pre(x, h, l, False)[0].ravel()
        e=metric.RMSE(x[:,target_idx][-l:].ravel(),pre)
        cnn_losses.append(e)
    best_h=hypers[cnn_losses.index(min(cnn_losses))]
    return best_h 
def rcv(hypers,x,target_idx,l):
    cnn_losses=[]
    for h in hypers:
        pre=rcnn_pre(x, h, l, False).ravel()
        e=metric.RMSE(x[:,target_idx][-l:].ravel(),pre)
        cnn_losses.append(e)
    best_h=hypers[cnn_losses.index(min(cnn_losses))]
    return best_h ,min(cnn_losses)
if __name__ == "__main__":
    data=pd.read_csv('VLCC.csv',index_col=0)
    start='Jan-96'
    end='Dec-20'
    data=data.loc[start:end].astype(float)
    variables=data.columns
   

   
    ratio=0.0001
    panamax=[]
    aframax=[]
    suezmax=[]
    capesize=[]
    for v in variables:
        if 'Panamax' in v:
            panamax.append(v)
        if 'Aframax' in v:
            aframax.append(v)
        if 'Suezmax' in v:
            suezmax.append(v)
        if 'Capesize' in v:
            capesize.append(v)
    panamax=[
           '75-77K DWT Panamax Bulkcarrier Newbuilding Prices',
        'Panamax 76K Bulkcarrier 5 Year Old Secondhand Prices',
        'Panamax Bulker Orderbook',
          '1 Year Timecharter Rate Panamax Bulkcarrier (Long Run Historical Series)',
            ]
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
    capesize=[
        '176-180K DWT Capesize Bulkcarrier Newbuilding Prices',
     'Capesize 5 Year Old Secondhand Prices (Long Run Historical Series)',
     'Capesize Bulker Orderbook',
       '1 Year Timecharter Rate Capesize Bulkcarrier (Long Run Historical Series)',
         ]
    metric=ForecastLib.TsMetric()
    ps_=np.arange(1,5)
    qs_=np.arange(5)
    ds_=[0,1]
    arima_hypers=list(product(ps_,qs_,ds_))
    rcvlosses=[]
    arima_errs=[]
    arima_bests=[]
    dif1st=False
    validation_l=36
    test_l=36
    testl=36
    vall=36
    step=1
    dif1st=False
    for features in [panamax,aframax,suezmax,capesize][:]:
        cnnpre = []
        target_idx=2
        target_f=features[target_idx]
        target=data[target_f].values.astype(np.float64)
        print(target_f)
        dat=data[features].values.astype(np.float64)
        x_=dat[:-testl,:]
        y_=target[:-testl].reshape(-1,1)

        orders=[6]
        scaler=preprocessing.MinMaxScaler()
        xscaler=preprocessing.MinMaxScaler()
        xscaler.fit(dat[:-testl,:])
        x_=xscaler.transform(dat)
            
        
        yscaler=preprocessing.MinMaxScaler()
        yscaler.fit(target[:-testl].reshape(-1,1))
        y_=yscaler.transform(target.reshape(-1,1))

        pools=[2]
        chs=[1,2,3,4]
        bs=[200]
        weight_decays=[0.00]
        lrs=[0.01]
        epochss=[1000]
        rnn_hs=[2,4,8,16]
    
        for ch in [1,2,3,4]:
            chs=[ch]

            cnn_hypers=list(product(chs,pools,orders,bs,weight_decays,lrs,epochss))

            cnn_best_h=cv(cnn_hypers,x_,target_idx,validation_l)
        for seed in np.arange(10): 
            setup_seed(seed)   
            scaler=preprocessing.MinMaxScaler()
            scaler.fit(x_) 
            norm_dif=scaler.transform(x_)
            target_scaler=preprocessing.MinMaxScaler()
            target_scaler.fit(x_[:-test_l,target_idx].reshape(-1,1))
            norm_pre,nnmodel=cnn_pre(norm_dif, cnn_best_h, test_l, False)
            df_pre=target_scaler.inverse_transform(norm_pre.reshape(-1,1))
            prediction=(df_pre+target[-1-test_l:-1].reshape(-1,1)).ravel()
            cnnpre.append(prediction)
        ADCpre = np.concatenate([np.reshape(arr, (-1, 1)) for arr in cnnpre], axis=1)
        ADCpre=pd.DataFrame(ADCpre)
        ADCpre.to_csv(target_f+'Step'+str(step)+'ADC.csv')