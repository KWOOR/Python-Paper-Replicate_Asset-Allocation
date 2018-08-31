# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 22:00:37 2018

@author: 우람
"""



import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt
import scipy as sp
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\nonmun\\CVAR')

price=pd.read_csv('cva.csv', index_col='DATE')
price.index=pd.DatetimeIndex(price.index)
ret=pd.DataFrame(columns=price.columns)
ret=price/price.shift(1)-1
monthly_ret=(ret+1).resample('M').prod()-1
#price=price.resample("M").mean()
#logret=np.log(ret+1)
#%% 기대수익률 시나리오 생성
y=pd.DataFrame(columns=price.columns)
y=price[500:len(price)-10].copy()
for i in range(0,len(price)-510):    #시간이 너무 오래걸리니까 전부 다 카피하지말고 조금만 잘라서 하자..
    buff=price[:500].copy()
    for j in range(500):
        buff.iloc[j:j+1]=price.iloc[i+500:i+501].values*price.iloc[i+j+10:i+j+11].values/price.iloc[i+j:i+j+1].values
    y.iloc[i:i+1]= (pd.DataFrame(buff.sum()/500).T).values

exp_ret=y/y.shift(1)-1
exp_ret=(exp_ret+1).resample('M').prod()-1  #월별 수익률로...

#%% CVAR 공식

def constraints(x):
    return pd.DataFrame(x).sum().values-1 
    
def CVAR(x):    # 99% CVAR 가정
    a=pd.DataFrame(x).T
    a.columns=exp_ret.columns
    b=pd.concat([a,a,a,a,a,a], axis=0)
    vol=(monthly_ret.iloc[i+18:i+24]*b.values).sum(axis=1).std()       # 6개월치 표준편차 계산
   # VaR=(exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-2.33*vol
    CVaR=((exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-(vol/(0.01*np.sqrt(2*np.pi)))*np.exp(-(2.33**2)/2)).values
    #VaR=exp_ret.iloc[i:i+1]*x[0]
    return 0.01+CVaR   # CVaR이 0.1보다 작아야하는 제약식

def RETURN(x):
    return -(exp_ret[i:i+1]*x.T).sum(axis=1).values#-(exp_ret[i:i+1].T.dropna().T*x).sum(axis=1)

weight=list()
for i in range(len(exp_ret)):
    #weight_buff=np.repeat(1/exp_ret.iloc[i:i+1].notnull().sum(axis=1).values,exp_ret.iloc[i:i+1].notnull().sum(axis=1).values)
    #lbound=np.repeat(0, exp_ret.iloc[i:i+1].notnull().sum(axis=1))
    #ubound=np.repeat(1, exp_ret.iloc[i:i+1].notnull().sum(axis=1))
    weight_buff=np.array(exp_ret.iloc[i:i+1]).T
    lbound=np.repeat(0,len(exp_ret.T))
    ubound=np.repeat(1,len(exp_ret.T))
    bnds=tuple(zip(lbound, ubound))
    const=({'type':'eq', 'fun': constraints}, {'type':'ineq', 'fun':CVAR})
    result=sp.optimize.minimize(RETURN,weight_buff,method='SLSQP', constraints=const, bounds=bnds)   #Nelder-Mead 방식으로하면, 제약식이 안 먹히긴 하지만 대략 1 비슷하게 나오고, Nan값 없음!
    weight.append(result.x)

CVaR_weight=pd.DataFrame(weight, columns=exp_ret.columns,index=exp_ret.index)


#%%

def constraints(x):
    return pd.DataFrame(x).sum().values-1 
    
weight=list()
for i in range(len(exp_ret)):
    #weight_buff=np.repeat(1/exp_ret.iloc[i:i+1].notnull().sum(axis=1).values,exp_ret.iloc[i:i+1].notnull().sum(axis=1).values)
    #lbound=np.repeat(0, exp_ret.iloc[i:i+1].notnull().sum(axis=1))
    #ubound=np.repeat(1, exp_ret.iloc[i:i+1].notnull().sum(axis=1))
    #weight_buff=np.repeat(1/len(exp_ret.iloc[i:i+1].T), len(exp_ret.iloc[i:i+1].T))
    weight_buff=np.array(exp_ret.iloc[i:i+1]).T
    lbound=np.repeat(0,len(exp_ret.T))
    ubound=np.repeat(1,len(exp_ret.T))
    bnds=tuple(zip(lbound, ubound))
    def CVAR(x):    # 99% CVAR 가정
        a=pd.DataFrame(x).T
        a.columns=exp_ret.columns
        b=pd.concat([a,a,a,a,a,a], axis=0)
        vol=(monthly_ret.iloc[i+18:i+24]*b.values).sum(axis=1).std()       # 6개월치 표준편차 계산
   # VaR=(exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-2.33*vol
        CVaR=((exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-(vol/(0.01*np.sqrt(2*np.pi)))*np.exp(-(2.33**2)/2)).values
    #VaR=exp_ret.iloc[i:i+1]*x[0]
        return 0.01+CVaR    # CVaR이 0.1보다 작아야하는 제약식

    def RETURN(x):
        return -(exp_ret[i:i+1]*x.T).sum(axis=1).values#-(exp_ret[i:i+1].T.dropna().T*x).sum(axis=1)

    const=({'type':'eq', 'fun': constraints}, {'type':'ineq', 'fun':CVAR})
    result=sp.optimize.minimize(RETURN,weight_buff,method='SLSQP', constraints=const, bounds=bnds)   #Nelder-Mead 방식으로하면, 제약식이 안 먹히긴 하지만 대략 1 비슷하게 나오고, Nan값 없음!
    weight.append(result.x)


CVaR_weight=pd.DataFrame(weight, columns=exp_ret.columns,index=exp_ret.index)

CVaR=pd.DataFrame()
CVaR['ret']=monthly_ret[23:].sum(axis=1)
CVaR['ret']=(CVaR_weight.shift(1).fillna(0).values*monthly_ret[23:].fillna(0).values).sum(axis=1)
CVaR['logret']=np.log(CVaR['ret']+1)
CVaR['cumret']=(CVaR['ret']+1).cumprod()-1
CVaR['logcumret']=CVaR['logret'].cumsum()

CVaR['logcumret'].plot()

CVaR['price']=monthly_ret.iloc[:,0].fillna(1)
CVaR['price'][0]=1
for i in range(len(CVaR)-1):
    CVaR['price'][i+1]=CVaR['price'].values[i]*(1+CVaR['ret'].values[i+1])

