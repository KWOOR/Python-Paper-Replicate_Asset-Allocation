# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 21:03:59 2018

@author: 우람
"""

import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt
import scipy as sp
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\nonmun\\AAA')

price=pd.read_excel('aaa_data.xlsx',index_col='DATE')
ret=pd.DataFrame(columns=price.columns)
ret=price/price.shift(1)-1

#%%

# Equal Weighted Portfolio

eqpf=pd.DataFrame()
eqpf['eqret']=ret.mean(axis=1)
eqpf['eqlogret']=np.log(eqpf['eqret']+1)
eqpf['eqcumret']=(eqpf['eqret']+1).cumprod()-1
eqpf['eqlogcumret']=eqpf['eqlogret'].cumsum()

eqpf['eqlogcumret'].plot()

eqpf['price']=ret.iloc[:,0].fillna(1)
for i in range(len(eqpf)-1):
    eqpf['price'][i+1]=eqpf['price'].values[i]*(1+eqpf['eqret'].values[i+1])

#%%
vol=pd.DataFrame(columns=price.columns)
vol=1/ret.rolling(window=3).std()

weight=vol.copy()
for i in range(len(vol)):
    weight[i:i+1]=vol[i:i+1]/float(vol[i:i+1].fillna(0).sum(axis=1))
weight.sum(axis=1)

#%% Volatility Weighted Portflio

vwpf=pd.DataFrame()
vwpf['vwret']=eqpf['eqret']
vwpf['vwret']=(weight.shift(1).fillna(0).values*ret.fillna(0).values).sum(axis=1)
vwpf['vwlogret']=np.log(vwpf['vwret']+1)
vwpf['vwcumret']=(vwpf['vwret']+1).cumprod()
vwpf['vwlogcumret']=vwpf['vwlogret'].cumsum()

vwpf['vwlogcumret'].plot()

vwpf['price']=ret.iloc[:,0].fillna(1)
for i in range(len(vwpf)-1):
    vwpf['price'][i+1]=vwpf['price'].values[i]*(1+vwpf['vwret'].values[i+1])
    
#%%  MoM EQ    mean을 median으로 바꾸면 딱 반절!!!

mom=np.log(ret+1).rolling(window=6, center=False).sum()
#로그에 오류가 나면 그냥 안 씌우고 돌려도... 차이 없음

mompf=pd.DataFrame()
mompf['ret']=mom.sum(axis=1)
for i in range(len(mom)-1):
    mompf['ret'][i+1]=(((mom.iloc[i:i+1] >mom.mean(axis=1)[i])/(int((mom.iloc[i:i+1] >mom.mean(axis=1)[i]).sum(axis=1)))).values*ret.iloc[i+1:i+2].fillna(0).values).sum(axis=1)
# 평균값하고 같은 값이 있을 수 있으니까... 부호를 부등호로 바꿔야 할 수도 있음!!

mompf['logret']=np.log(mompf['ret']+1)
mompf['cumret']=(mompf['ret']+1).cumprod()-1
mompf['logcumret']=mompf['logret'].cumsum()

mompf['logcumret'].plot()

mompf['price']=ret.iloc[:,0].fillna(1)
mompf['price'][1:7]=1
for i in range(len(mompf)-7):
    mompf['price'][i+7]=mompf['price'].values[i+6]*(1+mompf['ret'].values[i+7])
    
#%% MoM VW
momvwpf=mom.copy()
for i in range(len(mom)):
    momvwpf.iloc[i:i+1]=(mom.iloc[i:i+1] >mom.mean(axis=1)[i])*1
# 평균값하고 같은 값이 있을 수 있으니까... 부호를 부등호로 바꿔야 할 수도 있음!!
    
momvwpf=momvwpf*(1/ret.rolling(window=3).std())

weight=momvwpf.copy()
for i in range(len(vol)):
    weight[i:i+1]=momvwpf[i:i+1]/float(momvwpf[i:i+1].fillna(0).sum(axis=1))

momvwpf=pd.DataFrame()
momvwpf['ret']=mompf['ret']
momvwpf['ret']=(weight.shift(1).fillna(0).values*ret.fillna(0).values).sum(axis=1)
momvwpf['logret']=np.log(momvwpf['ret']+1)
momvwpf['cumret']=(momvwpf['ret']+1).cumprod()-1
momvwpf['logcumret']=momvwpf['logret'].cumsum()

momvwpf['logcumret'].plot()

momvwpf['price']=ret.iloc[:,0].fillna(1)
for i in range(len(momvwpf)-1):
    momvwpf['price'][i+1]=momvwpf['price'].values[i]*(1+momvwpf['ret'].values[i+1])
#%% min variance

def minvarpf(x):
    var=x.T@cov.fillna(0)@x
    sig=var**0.5
    return sig

def constraints(x):
    return x.sum()-1

def minvol(cov, lb, ub):
    x0=np.repeat(1/cov.shape[1], cov.shape[1])
    lbound=np.repeat(lb, cov.shape[1])
    ubound=np.repeat(ub, cov.shape[1])
    bnds=tuple(zip(lbound, ubound))
    const=({'type':'eq', 'fun': constraints})
    result=sp.optimize.minimize(minvarpf, x0, method='SLSQP', constraints=const, bounds=bnds)
    
    return result.x
    
#momvwpf=mom.copy()
#for i in range(len(mom)):
#    momvwpf.iloc[i:i+1]=(mom.iloc[i:i+1] >mom.mean(axis=1)[i])*1

minvar=pd.DataFrame()
minvar['ret']=ret.sum(axis=1)
for i in range(6,len(mom)-1):
    cov=(((mom.iloc[i:i+1]>mom.mean(axis=1)[i]).any()==True)*price[i-5:i+1]).replace(0,np.nan).dropna(axis=1).cov()
    weight_buff=minvol(cov,0,1)
    ret_buff=(((mom.iloc[i:i+1]>mom.mean(axis=1)[i]).any()==True)*ret[i+1:i+2]).replace(0,np.nan).dropna(axis=1)
    minvar['ret'][i+1:i+2]=(ret_buff*weight_buff).sum(axis=1)
    #평균하고 같은 값이 있을 수 있으므로 부호를 부등호로 바꿔야 할 수도 있음
minvar['ret'][0:7]=np.nan
minvar['logret']=np.log(minvar['ret']+1)
minvar['cumret']=(minvar['ret']+1).cumprod()-1
minvar['logcumret']=minvar['logret'].cumsum()

minvar['logcumret'].plot()

minvar['price']=ret.iloc[:,0].fillna(1)
minvar['price'][1:7]=1
for i in range(len(minvar)-7):
    minvar['price'][i+7]=minvar['price'].values[i+6]*(1+minvar['ret'].values[i+7])
    
#%% 누적 수익률 비교하기
plt.plot(eqpf['eqcumret'], label='EQ Cumret')
plt.plot(vwpf['vwcumret'], label='VW Cumret')
plt.plot(mompf['cumret'], label='MoM Cumret')
plt.plot(momvwpf['cumret'], label='MoMVW Cumret')
plt.plot(minvar['cumret'], label='MinVar Cumret')
plt.legend(loc='upper left')
plt.show()
    
    
    
    
    
    
    
    
    
    
    
