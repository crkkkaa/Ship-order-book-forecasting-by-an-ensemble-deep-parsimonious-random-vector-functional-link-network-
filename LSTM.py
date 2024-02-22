# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:55:49 2022

@author: ruobi
"""
import pandas as pd
import numpy as np
import torch
from basic import setup_seed,LSTM
from itertools import product
import ForecastLib
from sklearn import preprocessing
import matplotlib.pyplot as plt
from nbeats import TsMetric, setup_seed
plt.rcParams["font.family"] = "Times New Roman"
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
def rnn_format_data(x_,y_,order,step=1):
    n_sample=x_.shape[0]-order-step+1
    x=np.zeros((n_sample,order,x_.shape[1]))
    y=np.zeros((n_sample,y_.shape[1]))
    for i in range(n_sample):
        x[i,:,:]=x_[i:i+order,:]
        y[i,:]  =y_[i+order+step-1,:]
    return x,y
def generate_batches(x,y,batch_size,shuffle=False):
    """
    x (n_sample,n_channel,n_feature)
    y (n_sample,n_output)
    """
    n_sample=x.shape[0]
    n_batch=int(n_sample/batch_size)
    if shuffle:
        np.random.shuffle(x)
        np.random.shuffle(y)
    batchx,batchy=[],[]
    for i in range(n_batch):
        bx=x[i*batch_size:(i+1)*batch_size,:]
        by=y[i*batch_size:(i+1)*batch_size,:]
        batchx.append(torch.from_numpy(bx).float())
        batchy.append(torch.from_numpy(by).float())
        
    if n_batch*batch_size<n_sample:
        batchx.append(torch.from_numpy(x[n_batch*batch_size:,:]).float())
        batchy.append(torch.from_numpy(y[n_batch*batch_size:,:]).float())
    return batchx,batchy
def LSTM_pre(trainx,trainy,testx,hyper):
    order=hyper[0]
    hidden_size=hyper[1]
    nl=hyper[2]
    lr=hyper[3]
    batch_size=hyper[4]
    weight_decay=hyper[5]
    epochs=hyper[6]
    input_size=trainx.shape[-1]
    output_size=trainy.shape[-1]
    model=LSTM(input_size, hidden_size, output_size,num_layers=nl)
    batchx,batchy=generate_batches(trainx,trainy,batch_size,shuffle=False)
    n_batch=len(batchx)
    criterion = torch.nn.MSELoss(reduction='mean')    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    last_loss=0

    for epoch in range(epochs):
        rl=0
        n=0
        for i_batch in range(n_batch):
            x=batchx[i_batch].requires_grad_(True)            
            y=batchy[i_batch]                      
            outputs = model(x)
            optimizer.zero_grad()
            loss = criterion(outputs, y)    
            n+=1
            rl+=loss.item()                     
            loss.backward()  
            optimizer.step()
        if (epoch)%20==0:            
            last_loss=rl/n
    output=model(torch.tensor(testx).float()).detach().cpu().numpy()
    return output
def cv(x_,y_,pre_l):
    orders=[6]
    hidden_size=[2**int(i) for i in np.arange(1,5)]
    nls=[1,2]
    lrs=[0.1,0.01,0.001]
    batch_sizes=[200]
    weight_decay=[0]
    epochs=[100]
    losses=[]
    bhs=[]
    scaler=preprocessing.MinMaxScaler()
    scaler.fit(x_[:-pre_l,:])
    x_=scaler.transform(x_)
    for order in orders:
        allx,ally=rnn_format_data(x_,y_,order)
        trainx,trainy=allx[:-pre_l,:],ally[:-pre_l,:]
        testx,testy=allx[-pre_l:,:],ally[-pre_l:,:]
        hypers=list(product([order],hidden_size,nls,lrs,batch_sizes,weight_decay,epochs))
        loss=[]
        for i, h in enumerate(hypers):
            p=LSTM_pre(trainx,trainy,testx,h)
            e=compute_error(testy, p)
            loss.append(e['RMSE'])
        minl=min(loss)
        bhs.append(hypers[loss.index(minl)])
        losses.append(minl)
    bh=bhs[losses.index(min(losses))]
    return bh
            
            
            
if __name__ == "__main__":
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
       '1 Year Timecharter Rate Suezmax (Long Run Historical Series)'
     ]
    panamax=[
        '75-77K DWT Panamax Bulkcarrier Newbuilding Prices',
     'Panamax 76K Bulkcarrier 5 Year Old Secondhand Prices',
     'Panamax Bulker Orderbook',
       '1 Year Timecharter Rate Panamax Bulkcarrier (Long Run Historical Series)'
     ]
    capesize=[
        '176-180K DWT Capesize Bulkcarrier Newbuilding Prices',
     'Capesize 5 Year Old Secondhand Prices (Long Run Historical Series)',
     'Capesize Bulker Orderbook',
       '1 Year Timecharter Rate Capesize Bulkcarrier (Long Run Historical Series)',
    ]
    handysize=[
     '25-30K DWT Handysize Bulkcarrier Newbuilding Prices',
     'Handysize 37k dwt 5 Year Old Secondhand Prices',
     'Handysize Bulker Orderbook',
       '1 Year Timecharter Rate Handysize Bulkcarrier (Long Run Historical Series)',
     ]
    handymax=['61-63K DWT Handymax Bulkcarrier Newbuilding Prices',
     'Handymax 58k dwt 5 Year Old Secondhand Prices',
     'Handymax Bulker Orderbook',
     'Handymax Bulker Orderbook.1',
     'Handymax Bulker Orderbook.2',
     ]

    dif1st=False
    step=1
    for feats in [aframax,suezmax,panamax,capesize][:]:
        
        targf=feats[2]
        testl=36
        vall=36
        target=data[targf].values.astype(np.float64)
        dat=data[feats].values.astype(np.float64)
        
        x_=dat[:-testl,:]#
        y_=target[:-testl].reshape(-1,1)
        if dif1st:
            dif_dat=dat[1:-testl,:]-dat[:-testl-1,:]
            dif_target=target[1:-testl].reshape(-1,1)-target[:-testl-1].reshape(-1,1)

        if '/' in targf:
            targf=targf.replace('/',' ')
        xscaler=preprocessing.MinMaxScaler()
        xscaler.fit(dat[:-testl,:])
        x_=xscaler.transform(dat)
        
        yscaler=preprocessing.MinMaxScaler()
        yscaler.fit(target[:-testl].reshape(-1,1))
        y_=yscaler.transform(target.reshape(-1,1))
        validation_l=36
        test_l=36
        best_hyper=cv(x_,y_,validation_l)
        
        xscaler=preprocessing.MinMaxScaler()
        xscaler.fit(dat[:-test_l])
        normx=xscaler.transform(dat)

        
        yscaler=preprocessing.MinMaxScaler()
        yscaler.fit(target[:-test_l].reshape(-1,1))
        normy=yscaler.transform(target.reshape(-1,1))
        
        allx,ally=rnn_format_data(normx,normy,best_hyper[0])
        trainx,trainy=allx[:-test_l,:],ally[:-test_l,:]
        testx,testy=allx[-test_l:,:],ally[-test_l:,:]
        predictions = []
        for seed in np.arange(10):    
            np.random.seed(seed)            
            normp = LSTM_pre(trainx, trainy, testx, best_hyper).reshape(-1, 1)
            lstmp=yscaler.inverse_transform(normp.reshape(-1,1))
            predictions.append(lstmp)
        lstmpre=np.concatenate(predictions,axis=1)
        lstmpre=pd.DataFrame(lstmpre)
        lstmpre.to_csv(targf+'Step'+str(step)+'LSTM.csv')

        
        