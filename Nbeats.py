# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:15:58 2024

@author: crkkk
"""


import warnings
from itertools import product,combinations
import numpy as np
from sklearn import preprocessing
import pandas as pd
from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch
import matplotlib.pyplot as plt
import torch
import random
import warnings
warnings.filterwarnings("ignore")
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
warnings.filterwarnings(action='ignore', message='Setting attributes')
def compute_err(actual,prediction,history):
    actual=actual.ravel()
    prediction=prediction.ravel()
    history=history.ravel()
    metric=TsMetric()
    _mase=metric.MASE(actual, prediction, history)
    _rmse=metric.RMSE(actual, prediction)
    _mape=metric.MAPE(actual, prediction)
    _err={'MASE':_mase,'MAPE':_mape,'RMSE':_rmse}
    return _err
class TsMetric(object):
    def __init__(self):
        pass
    def RMSE(self,actual, pred):
        """
        RMSE = sqrt(1/n * sum_{i=1}^{n}{pred_i - actual_i} )
        input: actual and pred should be np.array
        output: RMSE
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        RMSE = np.sqrt( 1/len(actual) *np.linalg.norm(actual - pred,2)**2)
        return RMSE

    def MAE(self,actual, pred):
        '''
        MAE = 1/n * sum_{i=1}^{n}|pred_i - actual_i} |
        input: actual and pred should be np.array
        output: MAE

        '''
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        MAE =  1/len(actual) *np.linalg.norm(actual - pred,1)
        return MAE


    def MASE(self,actual, pred, history):
        '''
        MASE = 1/n * sum_{i=1}^{n}|pred_i - actual_i} |/ sum_traning(|diff|)
        input: actual and pred should be np.array
        output: MASE

        '''
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        MAE =  1/len(actual) *np.linalg.norm(actual - pred,1)
        Scale =  1/(len(history)-1) * np.linalg.norm(np.diff(history),1)
        MASE = MAE/Scale

        return MASE

    def MAPE(self,actual, pred):
        '''
        MAPE = 1/n * sum_{i=1}^{n} |pred_i - actual_i} |/|actual_i|
        input: actual and pred should be np.array
        output: MAPE

        '''
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        MAPE =  1/len(actual) *np.linalg.norm((actual - pred)/actual, 1)

        return MAPE

    def sMAPE(actual, pred):
        """
        1/n  *  SUM_{i=1 to n}  { ( |pred_i-actual_i|)   /  (0.5*|pred_i|+0.5*|actual_i|)}
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        sMAPE = 1/len(actual) * np.sum(2*np.abs(actual - pred)/(np.abs(actual)+np.abs(pred)))
        return sMAPE

    def RAE(actual, pred, compared):
        """
        INPUT: actual, pred, a prediction to be compared with\
        :return:
            l1(pred-actual)/ l1(compared - actual)
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred) == type(compared)) & (actual.shape == pred.shape ==compared.shape)
        nom = np.linalg.norm(actual-pred,1)
        denom = np.linalg.norm(actual - compared,1)
        return nom/denom

    def RSE(actual, pred, compared):
        """
        INPUT: actual, pred, a prediction to be compared with
        :return:
            l2(pred-actual)/ l2(compared - actual)
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred) == type(compared)) & (actual.shape == pred.shape ==compared.shape)
        nom = np.linalg.norm(actual-pred,2) ** 2
        denom = np.linalg.norm(actual - compared,2) ** 2
        return nom/denom

    def Corr(actual, pred):
        """
        :param actual: 
        :param pred: 
        return     np.dot(actual - mean(actual), pred -mean(pred)) / norm(actual-mean(acutal)*norm(pred- mean(pred))
        """
        nom = np.dot(actual -np.mean(actual), pred-np.mean(pred))
        denom =np.linalg.norm(actual-np.mean(actual)) * np.linalg.norm(pred- np.mean(pred))
        return nom/denom
def rnn_format_data(x_,y_,order):
    n_sample=x_.shape[0]-order
    x=np.zeros((n_sample,order*x_.shape[1]))
    y=np.zeros((n_sample,y_.shape[1]))
    for i in range(n_sample):
         x[i,:]=x_[i:i+order,:].ravel()
         y[i,:]  =y_[i+order,:]
    return x,y

def cv(x,y,val_l):
    nb_blocks_per_stack=[1,2]
    hidden_layer_units=[2,4]
    epoch=[200]
    batch_size=[4,16]
    hypers=list(product(nb_blocks_per_stack,hidden_layer_units,epoch,batch_size))
    error=[]
    metric=TsMetric()
    target=y[-val_l:]
    for hyper in hypers:
        p=Nbeat_pre(x,y,val_l,hyper)
        # print(p.shape,target.shape)
        e=metric.RMSE(p.ravel(),target.ravel())
        error.append(e)
    bh=hypers[error.index(min(error))]
    return bh
def Nbeat_pre(x,y,testl,hyper):
    time_steps=x.shape[1]
    output_steps=y.shape[1]
    num_samples=x.shape[0]
    nb_blocks_per_stack=hyper[0]
    hidden_layer_units=hyper[1]
    epoch=hyper[2]
    batch_size=hyper[3]
    setup_seed(0)   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_pytorch = NBeatsPytorch(device=device,
                                  backcast_length=time_steps, 
                                  forecast_length=output_steps,                            
                                    stack_types=(NBeatsPytorch.GENERIC_BLOCK, NBeatsPytorch.GENERIC_BLOCK),
                                  nb_blocks_per_stack=nb_blocks_per_stack,
                                  thetas_dim=(8,8), 
                                  share_weights_in_stack=True,
                                  hidden_layer_units=hidden_layer_units)
    
    model_pytorch.compile(loss='mse', optimizer='adam')
    x_train, y_train, x_test, y_test = x[:-testl,:], y[:-testl], x[-testl:,:], y[-testl:]
    test_size = testl

    model_pytorch.fit(x_train, y_train, validation_data=(x_test, y_test), 
                      epochs=epoch, batch_size=batch_size)

    predictions_pytorch_forecast = model_pytorch.predict(x_test)
    
    np.testing.assert_equal(predictions_pytorch_forecast.shape,
                            (test_size, model_pytorch.forecast_length))
    
    predictions_pytorch_backcast = model_pytorch.predict(x_test, return_backcast=True)
    
    np.testing.assert_equal(predictions_pytorch_backcast.shape,
                            (test_size, model_pytorch.backcast_length))

    return predictions_pytorch_forecast


if __name__ == '__main__':
    bhs=[]
    data=pd.read_csv('Data.csv',index_col=0)
    start='Jan-96'
    end='Dec-20'
    data=data.loc[start:end].astype(float)
    var_name=data.columns

    suezmax=[]
    panamax=[]
    aframax=[]
    capesize=[]
    seeds=3
    for v in var_name:

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
        
        target=data[targf].values.astype(np.float64)
        data_=data[feats].values.astype(np.float64)   
        x_=data_[:-testl,:]
        y_=target[:-testl].reshape(-1,1)
        vall=36
        validation_l=vall
        if dif1st:
            dif_data_=data_[1:-testl,:]-data_[:-testl-1,:]
            dif_target=target[1:-testl].reshape(-1,1)-target[:-testl-1].reshape(-1,1)
    
        if '/' in targf:
            targf=targf.replace('/',' ')   
          
        target_idx=-1
        dif_data_=data_[1:-testl,:]-data_[:-testl-1,:]
        metric=TsMetric()
        order=6           
        xscaler=preprocessing.MinMaxScaler()
        xscaler.fit(data_[:-testl])
        normx=xscaler.transform(data_)
            
        yscaler=preprocessing.MinMaxScaler()
        yscaler.fit(target[:-testl].reshape(-1,1))
        normy=yscaler.transform(target.reshape(-1,1))
        pres=np.zeros((testl,order))

        predictions = []
        for seed in np.arange(10):
            np.random.seed(seed)            
            allx,ally=rnn_format_data(normx,normy,order)
            trainx,trainy=allx[:-testl,:],ally[:-testl]
            valx,valy=allx[-testl-validation_l:-testl,:],ally[-validation_l-testl:-testl,:]                               
            best_hyper=cv(allx[:-testl],ally[:-testl],validation_l)
            normp=Nbeat_pre(allx,ally,testl,best_hyper)
            nbeatspre=yscaler.inverse_transform(normp.reshape(-1,1))
            predictions.append(nbeatspre)
        nbeatspre=np.concatenate(predictions,axis=1)
        nbeatspre=pd.DataFrame(nbeatspre)
        nbeatspre.to_csv(targf+'Step'+str(step)+'Nbeats.csv')