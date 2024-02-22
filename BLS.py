#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:06:49 2023

@author: hello
"""
from easydict import EasyDict
import ForecastLib
import numpy as np
import pandas as pd
from nbeats import TsMetric, setup_seed
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from BLS_Regression import bls_regression
import ForecastLib
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
def get_swap_dict(d):
    return {v: k for k, v in d.items()}
def get_naive(ts_data,order,step):
    """
    ts_data: n_time*n_features
    assume the first column is the target
    """
    time,n_feature=ts_data.shape[0],ts_data.shape[1]
    n_sample=time-order-step+1
    y=np.zeros((n_sample,step))
    for i in range(n_sample):
        n=ts_data[i+order-1,0]
        y[i,:]=n
    return y
def format_data_flatten(ts_data,order,step):
    """
    ts_data: n_time*n_features
    assume the first column is the target
    """
    time,n_feature=ts_data.shape[0],ts_data.shape[1]
    n_sample=time-order-step+1
    x=np.zeros((n_sample,order*n_feature))
    y=np.zeros((n_sample,step))
    for i in range(n_sample):
        x[i,:]=ts_data[i:i+order,:].ravel()
        y[i,:]=ts_data[i+order:i+order+step,0]
    return x,y
def BLS_pre(BLSX_trv,BLSy_trv ,Xtest,attr):
    input_size=BLSX_trv.shape[-1]
    output_size=step
   
    s=attr.s
    C=attr.C
    NumFea=attr.NumFea
    NumWin=attr.NumWin
    NumEnhan=attr.NumEnhan
    results=bls_regression(BLSX_trv,BLSy_trv ,Xtest,norm_evaly,s,C,NumFea,NumWin,NumEnhan)
    ky_hat=results[-1]
    return ky_hat
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
    for feats in [panamax,aframax,suezmax,capesize][:]:
       
        targf=feats[2]
        print(targf)
                
        testl=36
        vall=36
        target=data[targf].values.astype(np.float64)
        dat=data[feats].values.astype(np.float64)
        
        x_=dat[:-testl,:]#-
        y_=target[:-testl].reshape(-1,1)
        if dif1st:
            dif_dat=dat[1:-testl,:]-dat[:-testl-1,:]
            dif_target=target[1:-testl].reshape(-1,1)-target[:-testl-1].reshape(-1,1)

        if '/' in targf:
            targf=targf.replace('/',' ')
        
        
        ts_data=dat
        """5 fold cross validation"""
        max_ts=ts_data[:-testl].max(axis=0)
        min_ts=ts_data[:-testl].min(axis=0)
        print(ts_data)
        max_y,min_y=max_ts[2],min_ts[2]
        den_ts=max_ts-min_ts
        ts_norm=(ts_data-min_ts[None,:])/den_ts[None,:]        
        orders=[6]
        s=[1]
        Cs=[2**-8,2**-6,2**-4,2**-2,0,1,2**2,2**4]
        NumFeas=[1,2,4]
        NumWins=[2,4,6]
        NumEnhans=[2,4,6]
        
        min_loss=np.inf
        setup_seed(0)
        for order in orders:
            allx,ally=format_data_flatten(ts_norm,order,step)
            train_l=allx.shape[0]-testl
            norm_trainx,norm_trainy=allx[:train_l],ally[:train_l]
        
            kf = KFold(n_splits=5)
            order_loss=[]
            for C in Cs:
                for NumFea in NumFeas:
                    for NumWin in NumWins:
                        for NumEnhan in NumEnhans[:1]:
                            
                            loss=0
                            attr=EasyDict()
                            attr['order']=order
                            attr['C']=C
                            attr['s']=1
                            attr['NumFea']=NumFea
                            attr['NumEnhan']=NumEnhan
                            attr['NumWin']=NumWin
                            for i, (train_index, val_index) in enumerate(kf.split(norm_trainx)):
                                k_trainx,k_trainy=norm_trainx[train_index,:],norm_trainy[train_index,:]
                                k_valx,k_valy=norm_trainx[val_index,:],norm_trainy[val_index,:]                            
                                s=attr.s
                                C=attr.C
                                NumFea=attr.NumFea
                                NumWin=attr.NumWin
                                NumEnhan=attr.NumEnhan
                                
                                results=bls_regression(k_trainx,k_trainy ,k_valx,k_valy,s,C,NumFea,NumWin,NumEnhan)
                                ky_hat=results[-1].A
                                
                                err=compute_error(ky_hat.ravel(),k_valy.ravel())['RMSE']
                                loss+=err
                            if loss<min_loss:
                                min_loss=loss
                                best_attr=attr
            import pickle
            with open('Hyper/BLS'+targf+'.pickle', 'wb') as handle:
               pickle.dump(best_attr, handle, protocol=pickle.HIGHEST_PROTOCOL)
            blspre=[]
            for seed in np.arange(10):
                np.random.seed(seed)
                best_attr.randseed=seed            
                train_l=allx.shape[0]-testl
                norm_evalx,norm_evaly=allx[train_l:],ally[train_l:]
                s=best_attr.s
                C=best_attr.C
                NumFea=best_attr.NumFea
                NumWin=best_attr.NumWin
                NumEnhan=best_attr.NumEnhan
            
                results=bls_regression(norm_trainx,norm_trainy ,norm_evalx,norm_evaly,s,C,NumFea,NumWin,NumEnhan)
                ytest_hat_norm=results[-1]              
                ytest_hat=ytest_hat_norm*(max_y-min_y)+min_y        
                ytest=norm_evaly*(max_y-min_y)+min_y
                blspre.append(ytest_hat)
            blspre=np.concatenate(blspre,axis=1)
            blspre=pd.DataFrame(blspre)
            blspre.to_csv(targf+'Step'+str(step)+'BLS.csv')

           