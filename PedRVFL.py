# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 08:56:04 2022

@author: ruobi
"""
from DeepRVFL_.DeepRVFL import DeepRVFL
from utils import MSE, config_MG, load_MG
from sklearn import preprocessing
from DeepRVFL_.DeepRVFL import DeepRVFL
import pandas as pd
import numpy as np
from itertools import product,combinations
from sklearn.linear_model import Ridge,Lasso
import ForecastLib
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
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
    return  x.T,y.T

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
class Struct(object): pass
def config_load(iss,IP_indexes):
    configs = Struct()        
    configs.iss = iss     
    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0 
    configs.enhConf = Struct()
    configs.enhConf.connectivity = 1 
    
    configs.readout = Struct()
    configs.readout.trainMethod = 'Ridge'
    
    return configs 
def select_indexes(data, indexes):

    # if len(data) == 1:
    return data[:,indexes]
def dRVFL_predict(hyper,data,train_idx,test_idx,layer,s,last_states=None):
    np.random.seed(s)
    Nu=datastr.inputs.shape[0]
    Nh = hyper[0][0] 
    Nl = layer    
    reg=[]
    iss=[]
    for h in hyper:
        reg.append( h[1])        
        iss.append(h[2])
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nh, Nl, configs)
    train_targets = select_indexes(datastr.targets, train_idx)
    if Nl==1:
        
        states = deepRVFL.computeLayerState(0,datastr.inputs)
    else:
        
       
        states=deepRVFL.computeLayerState(Nl-1,datastr.inputs,last_states[:,:])
    train_states = select_indexes(np.concatenate([states,datastr.inputs],axis=0), train_idx)
    test_states = select_indexes(np.concatenate([states,datastr.inputs],axis=0), test_idx)
    deepRVFL.trainReadout(train_states[:,:], train_targets, reg[-1])
 
    test_outputs_norm = deepRVFL.computeOutput(test_states[:,:]).T

    return test_outputs_norm,states[:,:]
def stackedRVFL_predict(hyper,data,train_idx,test_idx,s,alpha,gamma):
    np.random.seed(s)
    Nu=datastr.inputs.shape[0]
    Nr = hyper[0][0] 
    Nl = len(hyper)
    reg=[]   
    iss=[]
    for h in hyper:
        reg.append( h[1])
       
        iss.append(h[2])
    
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nr, Nl, configs)
    last_states=None
    outputs=np.zeros((len(test_idx),Nl))
    trainpres=np.zeros((len(train_idx),Nl))
    train_targets = select_indexes(datastr.targets, train_idx)
    for l in range(Nl):
        if l==0:
            states = deepRVFL.computeLayerState(l,datastr.inputs,inistate=None)
        else:
            states=deepRVFL.computeLayerState(l,datastr.inputs,last_states)
        last_states=states
        train_states = select_indexes(np.concatenate([states,datastr.inputs],axis=0), train_idx)
        train_targets = select_indexes(datastr.targets, train_idx)
        test_states = select_indexes(np.concatenate([states,datastr.inputs],axis=0), test_idx)
        deepRVFL.trainReadout(train_states, train_targets, reg[l])
        test_outputs_norm = deepRVFL.computeOutput(test_states).T
        outputs[:,l:l+1]=test_outputs_norm
        train_outputs_norm = deepRVFL.computeOutput(train_states).T

        trainpres[:,l:l+1]=train_outputs_norm
    ensemble=KernelRidge(alpha=alpha,gamma=gamma,kernel='rbf')
    ensemble.fit(trainpres,train_targets.T)
    dyn_p=ensemble.predict(outputs)
    return np.median(outputs,axis=1).reshape(-1,1),outputs,dyn_p
def ensemble(trainx,trainy,testx,hyper,prel):
    alpha=hyper[0]
    gamma=hyper[1]
    m=KernelRidge(alpha=alpha,gamma=gamma,kernel='linear')
    m.fit(trainx,trainy)
    p=m.predict(testx)
    return p
def edRVFL_predict(hyper,data,train_idx,test_idx,s):
    np.random.seed(s)
    Nu=datastr.inputs.shape[0]
    Nr = hyper[0][0] 
    Nl = len(hyper) 
    reg=[]
   
    iss=[]
    for h in hyper:
        reg.append( h[1])
       
        iss.append(h[2])
    
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nr, Nl, configs)
    last_states=None
    outputs=np.zeros((len(test_idx),Nl))
    trainpres=np.zeros((len(train_idx),Nl))
    train_targets = select_indexes(datastr.targets, train_idx)
    for l in range(Nl):
        if l==0:

            states = deepRVFL.computeLayerState(l,datastr.inputs,inistate=None)

        else:
            states=deepRVFL.computeLayerState(l,datastr.inputs,last_states)
        last_states=states
        train_states = select_indexes(np.concatenate([states,datastr.inputs],axis=0), train_idx)
        train_targets = select_indexes(datastr.targets, train_idx)
        test_states = select_indexes(np.concatenate([states,datastr.inputs],axis=0), test_idx)
        deepRVFL.trainReadout(train_states, train_targets, reg[l])
        test_outputs_norm = deepRVFL.computeOutput(test_states).T
        outputs[:,l:l+1]=test_outputs_norm
        train_outputs_norm = deepRVFL.computeOutput(train_states).T

        trainpres[:,l:l+1]=train_outputs_norm
    dyn_p=[]
    test_targets = select_indexes(datastr.targets, test_idx)
    _error=abs(trainpres[-1:,:]-train_targets[0][-1])
    w=np.power(_error,-1) / np.sum(np.power(_error,-1))
    dyn_p.append(w.dot(outputs[:1].T))
    for i in range(len(test_idx)-1):
        _error=abs(outputs[i:i+1,:]-test_targets[0][i])
        w=np.power(_error,-1) / np.sum(np.power(_error,-1))
        dyn_p.append(w.dot(outputs[i+1:i+2,:].T))
    return np.median(outputs,axis=1).reshape(-1,1),outputs,np.array(dyn_p),trainpres,train_targets.T,outputs#.mean(axis=1).reshape(-1,1)

def RVFL_predict(hyper,data,train_idx,test_idx,s):

    np.random.seed(s)
    Nu=1
    Nh = hyper[0] 
    Nl = 1
    reg = hyper[1]
    iss=hyper[2]
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nh, Nl, configs)
    states = deepRVFL.computeState(datastr.inputs, deepRVFL.IPconf.DeepIP)  
#    print(transient,states[0].shape,states[0][0,:2],states[0][0,-2:])              
    train_states = select_indexes(states, train_idx)
    train_targets = select_indexes(datastr.targets, train_idx)
    test_states = select_indexes(states, test_idx)
    outputs=np.zeros((len(test_idx),Nl))
    for i in range(Nl):              
        deepRVFL.trainReadout(train_states[i*Nh:i*Nh+Nh,:], train_targets, reg)
        test_outputs_norm = deepRVFL.computeOutput(test_states[i*Nh:i*Nh+Nh,:]).T
        outputs[:,i:i+1]=test_outputs_norm
    return np.median(outputs,axis=1).reshape(-1,1)
def cross_validation(data,raw_data,train_idx,val_idx,Nl,scaler=None,s=0,boat=50):
    best_hypers=[]
    np.random.seed(s)
    layer_s=None
    totall=len(train_idx)+len(val_idx)
    total_idx=np.arange(totall)
    splits=np.array_split(total_idx,Nl)
    for i in range(Nl):
        layer=i+1
        layer_h,layer_s=layer_cross_validation(data,raw_data,train_idx,val_idx,layer,
                            scaler=scaler,s=s,last_states=layer_s,best_hypers=best_hypers.copy(),boat=boat)
        Nhs=[layer_h[0]]     
        best_hypers.append(layer_h)
    
   
    return best_hypers 
def layer_cross_validation(data,raw_data,train_idx,val_idx,layer,
                           scaler=None,s=0,last_states=None,best_hypers=None,boat=50):
    cvloss=[]
    np.random.seed(s)
    states=[]
    if layer==1:
        Nhs=[1,2,4,8,16]
        regs=[0.001,0.0001,0.00001,0]
        input_scales=[1,2,0.1,0.01,0.0001]
    else:
        Nhs=[1,2,4,8,16]
        regs=[0.001,0.0001,0.00001,0]
        input_scales=[1]
    best_loss=np.inf
    for Nh in Nhs[:]:
        for reg in regs[:]:
            for iss in input_scales:
                ar={'layer': layer,'data': data,'raw_data': raw_data,'last_states': last_states,
                    'scaler':scaler,'s':s,'val_idx':val_idx,'train_idx':train_idx,
                    'best_hypers':best_hypers,'Nhs':Nh,'regs':reg,'input_scale':iss}
                closs=layer_obj(ar)
                if closs<best_loss:
                    best_loss=closs
                    args=ar
    
    best_hyper=[args['Nhs'],args['regs'],args['input_scale']]

    if layer>1:

            hyper_=best_hypers.copy()
            hyper_.append(best_hyper)

    else:

        hyper_=[best_hyper]
    _,best_state=dRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                         s,last_states=last_states)

    return best_hyper,best_state 
def layer_obj(args):
    layer=args['layer']
    best_hypers=args['best_hypers']
    hyper=[args['Nhs'],args['regs'],args['input_scale']]
    data=args['data']
    train_idx,val_idx=args['train_idx'],args['val_idx']
    scaler=args['scaler']
    s=args['s']
    raw_data,last_states=args['raw_data'],args['last_states']
    if layer>1:

            hyper_=[i for i in best_hypers]
            hyper_.append(hyper)
    else:
        hyper_=[hyper]
    e=0
    for se in range(seeds):
        test_outputs_norm,_=dRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                          se,last_states=last_states)
        test_outputs=scaler.inverse_transform(test_outputs_norm)
        actuals=raw_data[-len(val_idx):]
        test_err=compute_error(actuals,test_outputs,None)['MAE'] 
        e+=test_err
    return e       
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
for feats in [panamax][:]:
    seeds = 10  
    all_predictions = []
    for seed in range(seeds):    
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
        if '/' in targf:
            targf=targf.replace('/',' ')
        name=targf+'Hyperstep'+str(step)+'.pkl'

        f=open(name,'rb')
        pimsebh=pickle.load(f)
        f.close()
        print(pimsebh)
        order=pimsebh[0]

        xscaler=preprocessing.MinMaxScaler()
        xscaler.fit(dat[:-testl,:])
        normx_=xscaler.transform(dat)
        
        yscaler=preprocessing.MinMaxScaler()
        yscaler.fit(target[:-testl].reshape(-1,1))
        normy_=yscaler.transform(target.reshape(-1,1))
        
        datastr=Struct()
        datastr.inputs,datastr.targets=format_disc_data(normx_,normy_,order,max_lag=12,step=1)
        train_l=datastr.inputs.shape[1]-vall-testl
        train_idx=range(train_l)
        val_idx=range(train_l,train_l+vall)
        test_idx=range(train_l+vall,datastr.inputs.shape[1])
        Nls=[1,2,4,8,10]
        for Nl in Nls:
           
            best_hypers=cross_validation(data,target[:-testl],train_idx,val_idx,Nl,scaler=yscaler,s=0,boat=50)
        print(best_hypers)
        
        
        _,_,_,trainpres,train_targets,valpres=edRVFL_predict(best_hypers,data,train_idx,val_idx,1)

      

        

    
