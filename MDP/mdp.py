# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 22:14:35 2018

@author: 우람
"""

import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt
import scipy as sp
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\nonmun\\MDP')

price=pd.read_excel('aaa_data.xlsx',index_col='DATE')
ret=pd.DataFrame(columns=price.columns)
ret=price/price.shift(1)-1

def MDD(d):
    x=np.maximum.accumulate(d)
    z=((x-d)/x).max(axis=0)
    return z

def constraints(x):
    return x.sum()-1 

    
weight=list()
for i in range(6,len(ret)):
    weight_buff=np.repeat(1/ret.iloc[i:i+1].notnull().sum(axis=1),ret.iloc[i:i+1].notnull().sum(axis=1) )
    lbound=np.repeat(0, ret.iloc[i:i+1].notnull().sum(axis=1))
    ubound=np.repeat(1, ret.iloc[i:i+1].notnull().sum(axis=1))
    bnds=tuple(zip(lbound, ubound))
    const=({'type':'eq', 'fun': constraints})
    def DDR(x):
        if len(ret.iloc[i-6:i].std().dropna())<len(x):
            avgvol=(ret.iloc[i-6:i].std().fillna(0).values*x).sum()   #6개월치 표준편차를 계산했음
        else:
            avgvol=(ret.iloc[i-6:i].std().dropna().values*x).sum()
            
        pfvol=(ret.iloc[i-6:i]*x[0]).sum(axis=1).std()               #마찬가지로 6개월치 표준편차 계산
        return pfvol/avgvol # pfvol/avgvol  #avgvol/pfvol의 최대화를 하기 위해 역수인 pfvol/avgvol의 최소화를 구함
    result=sp.optimize.minimize(DDR,weight_buff,method='SLSQP', constraints=const, bounds=bnds)   #Nelder-Mead 방식으로하면, 제약식이 안 먹히긴 하지만 대략 1 비슷하게 나오고, Nan값 없음!
    weight.append(result.x)
    #weight.iloc[i:i+1]=result.x

MDP_weight=pd.DataFrame(weight, columns=ret.columns,index=ret[6:].index  )
buff=pd.DataFrame(columns=ret.columns, index=ret[:6].index)
MDP_weight=pd.concat([buff,MDP_weight])

MDP=pd.DataFrame()
MDP['ret']=ret.sum(axis=1)
MDP['ret']=(MDP_weight.shift(1).fillna(0).values*ret.fillna(0).values).sum(axis=1)
MDP['logret']=np.log(MDP['ret']+1)
MDP['cumret']=(MDP['ret']+1).cumprod()-1
MDP['logcumret']=MDP['logret'].cumsum()

MDP['logcumret'].plot()

MDP['price']=ret.iloc[:,0].fillna(1)
for i in range(len(MDP)-1):
    MDP['price'][i+1]=MDP['price'].values[i]*(1+MDP['ret'].values[i+1])
    
print('MDP Return의 Sharpe:', (MDP['ret'].mean()*12-0.02)/(MDP['ret'].std()*(12**(1/2))))
print('MDP Return의 MDD:', MDD(MDP['price']))


  
    
#
#def DDR(x,i):
#    if len(ret.iloc[i-6:i].std().dropna())<len(x):
#        avgvol=(ret.iloc[i-6:i].std().fillna(0).values*x).sum()
#    else:
#        avgvol=(ret.iloc[i-6:i].std().dropna().values*x).sum()
#        
#    pfvol=(ret.iloc[i-6:i]*x[0]).sum(axis=1).std()
#    return avgvol/pfvol
#
#weight=list()
#for i in range(6,len(ret)):
#    weight_buff=np.repeat(1/ret.iloc[i:i+1].notnull().sum(axis=1),ret.iloc[i:i+1].notnull().sum(axis=1) )
#    lbound=np.repeat(0, ret.iloc[i:i+1].notnull().sum(axis=1))
#    ubound=np.repeat(1, ret.iloc[i:i+1].notnull().sum(axis=1))
#    bnds=tuple(zip(lbound, ubound))
#    const=({'type':'eq', 'fun': constraints})
#    result=sp.optimize.minimize(DDR,weight_buff,i,method='SLSQP', constraints=const, bounds=bnds)
#    weight.append(result.x)
#    #weight.iloc[i:i+1]=result.x
#
#a=pd.DataFrame(weight) #Nelder-Mead 방식으
#
#        
    
    
    


