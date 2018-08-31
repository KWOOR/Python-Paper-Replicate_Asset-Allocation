# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 19:48:22 2018

@author: 우람
"""

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

price=pd.read_excel('emp_price_20180614.xlsx', index_col='DATE')
ret=pd.DataFrame(columns=price.columns)
ret=price/price.shift(1)-1
#ret1=(ret+1).resample('M').cumprod()-1
ret=(ret+1).resample('M').prod()-1
price=price.resample('M').mean()
#%%

# Equal Weighted Portfolio

eqpf=pd.DataFrame()
eqpf['eqret']=ret.mean(axis=1)
eqpf['eqlogret']=np.log(eqpf['eqret']+1)
eqpf['eqcumret']=(eqpf['eqret']+1).cumprod()-1
eqpf['eqlogcumret']=eqpf['eqlogret'].cumsum()

eqpf['eqlogcumret'].plot()

eqpf['price']=ret.iloc[:,0].fillna(1)
eqpf['price'][0]=1
for i in range(len(eqpf)-1):
    eqpf['price'][i+1]=eqpf['price'].values[i]*(1+eqpf['eqret'].values[i+1])

#%%
vol=pd.DataFrame(columns=price.columns)
vol=1/price.rolling(window=3).std()

weight=vol.copy()
for i in range(len(vol)):
    weight[i:i+1]=vol[i:i+1]/float(vol[i:i+1].fillna(0).sum(axis=1))
weight.sum(axis=1)

#%% Volatility Weighted Portflio

vwpf=pd.DataFrame()
vwpf['vwret']=eqpf['eqret']
vwpf['vwret']=(weight.shift(1).fillna(0).values*ret.fillna(0).values).sum(axis=1)
vwpf['vwlogret']=np.log(vwpf['vwret']+1)
vwpf['vwcumret']=(vwpf['vwret']+1).cumprod()-1
vwpf['vwlogcumret']=vwpf['vwlogret'].cumsum()

vwpf['vwlogcumret'].plot()

vwpf['price']=ret.iloc[:,0].fillna(1)
vwpf['price'][0]=1
for i in range(len(vwpf)-1):
    vwpf['price'][i+1]=vwpf['price'].values[i]*(1+vwpf['vwret'].values[i+1])
    
#%%  MoM EQ    mean을 median으로 바꾸면 딱 반절!!!

mom=ret.rolling(window=6, center=False).sum()
mompf=pd.DataFrame()
mompf['ret']=mom.sum(axis=1)
for i in range(len(mom)-1):
    mompf['ret'][i+1]=(((mom.iloc[i:i+1] >=mom.mean(axis=1)[i])/(int((mom.iloc[i:i+1] >=mom.mean(axis=1)[i]).sum(axis=1)))).values*ret.iloc[i+1:i+2].fillna(0).values).sum(axis=1)

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
    momvwpf.iloc[i:i+1]=(mom.iloc[i:i+1] >=mom.mean(axis=1)[i])*1
momvwpf=momvwpf*(1/price.rolling(window=3).std())

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
momvwpf['price'][0]=1
for i in range(len(momvwpf)-1):
    momvwpf['price'][i+1]=momvwpf['price'].values[i]*(1+momvwpf['ret'].values[i+1])
    
    

#%%MOMVW 수정
buff=1/price.rolling(window=3).std()
for i in range(len(weight)):
    if weight.iloc[i,0]+weight.iloc[i,1]<0.1:
        number=0.1-weight.iloc[i,0]-weight.iloc[i,1]
        weight=weight.replace(weight.iloc[i,:].max(axis=0),weight.iloc[i,:].max(axis=0)-number)
        weight.iloc[i,0]+=number*(buff.iloc[i,0]/(buff.iloc[i,0]+buff.iloc[i,1]))
        weight.iloc[i,1]+=number*(buff.iloc[i,1]/(buff.iloc[i,0]+buff.iloc[i,1]))

for i in range(len(weight)):
    if weight.iloc[i,2]+weight.iloc[i,3]+weight.iloc[i,4]+weight.iloc[i,5]<0.1:
        number=0.1-weight.iloc[i,2]-weight.iloc[i,3]-weight.iloc[i,4]-weight.iloc[i,5]
        weight=weight.replace(weight.iloc[i,:].max(axis=0),weight.iloc[i,:].max(axis=0)-number)
        weight.iloc[i,2]+=number*(buff.iloc[i,2]/(buff.iloc[i,2]+buff.iloc[i,3]+buff.iloc[i,4]+buff.iloc[i,5]))
        weight.iloc[i,3]+=number*(buff.iloc[i,3]/(buff.iloc[i,2]+buff.iloc[i,3]+buff.iloc[i,4]+buff.iloc[i,5]))
        weight.iloc[i,4]+=number*(buff.iloc[i,4]/(buff.iloc[i,2]+buff.iloc[i,3]+buff.iloc[i,4]+buff.iloc[i,5]))
        weight.iloc[i,5]+=number*(buff.iloc[i,5]/(buff.iloc[i,2]+buff.iloc[i,3]+buff.iloc[i,4]+buff.iloc[i,5]))

for i in range(len(weight)):
    if weight.iloc[i,6]+weight.iloc[i,7]+weight.iloc[i,8]<0.2:
        number=0.2-weight.iloc[i,6]-weight.iloc[i,7]-weight.iloc[i,8]
        weight=weight.replace(weight.iloc[i,:].max(axis=0),weight.iloc[i,:].max(axis=0)-number)
        weight.iloc[i,6]+=number*(buff.iloc[i,6]/(buff.iloc[i,6]+buff.iloc[i,7]+buff.iloc[i,8]))
        weight.iloc[i,7]+=number*(buff.iloc[i,7]/(buff.iloc[i,6]+buff.iloc[i,7]+buff.iloc[i,8]))
        weight.iloc[i,8]+=number*(buff.iloc[i,8]/(buff.iloc[i,6]+buff.iloc[i,7]+buff.iloc[i,8]))
    if weight.iloc[i,8]<0.05:
        number=0.05-weight.iloc[i,8]
        weight=weight.replace(weight.iloc[i,:].max(axis=0),weight.iloc[i,:].max(axis=0)-number)
        weight.iloc[i,8]+=0.05-weight.iloc[i,8]

for i in range(len(weight)):
    if weight.iloc[i,10]+weight.iloc[i,9]<0.05:
        number=0.05-weight.iloc[i,10]-weight.iloc[i,9]
        weight=weight.replace(weight.iloc[i,:].max(axis=0),weight.iloc[i,:].max(axis=0)-number)
        weight.iloc[i,10]+=number*(buff.iloc[i,10]/(buff.iloc[i,10]+buff.iloc[i,9]))
        weight.iloc[i,9]+=number*(buff.iloc[i,9]/(buff.iloc[i,10]+buff.iloc[i,9]))
        
for i in range(len(weight)):
    if weight.iloc[i,0]+weight.iloc[i,1]>0.4:
        number=weight.iloc[i,0]+weight.iloc[i,1]-0.4
        weight=weight.replace(min(weight.iloc[i,0],weight.iloc[i,1]), min(weight.iloc[i,0],weight.iloc[i,1])-number)
        weight.iloc[i,14]+=number
    if weight.iloc[i,0]>0.4:
        number=weight.iloc[i,0]-0.4
        weight=weight.replace(weight.iloc[i,0], weight.iloc[i,0]-number)
        weight.iloc[i,14]+=number
    if weight.iloc[i,1]>0.2:
        number=weight.iloc[i,1]-0.2
        weight=weight.replace(weight.iloc[i,1], weight.iloc[i,1]-number)
        weight.iloc[i,14]+=number


for i in range(len(weight)):    #최댓값에서.. 그럴 일은 없겠지만 30%씩 빼는 경우도 있을것같음!!! 수정!!
    if weight.iloc[i,2]+weight.iloc[i,3]+weight.iloc[i,4]+weight.iloc[i,5]>0.4:
        number=weight.iloc[i,2]+weight.iloc[i,3]+weight.iloc[i,4]+weight.iloc[i,5]-0.4
        weight=weight.replace(max(weight.iloc[i,2],weight.iloc[i,3],weight.iloc[i,4],weight.iloc[i,5]),max(weight.iloc[i,2],weight.iloc[i,3],weight.iloc[i,4],weight.iloc[i,5])-number)
        weight.iloc[i,14]+=number
    if min(weight.iloc[i,2],weight.iloc[i,3],weight.iloc[i,4],weight.iloc[i,5])<0:
        number=min(weight.iloc[i,2],weight.iloc[i,3],weight.iloc[i,4],weight.iloc[i,5])
        weight=weight.replace(min(weight.iloc[i,2],weight.iloc[i,3],weight.iloc[i,4],weight.iloc[i,5]), min(weight.iloc[i,2],weight.iloc[i,3],weight.iloc[i,4],weight.iloc[i,5])-number)
        weight.iloc[i,14]+=number
    if weight.iloc[i,2]>0.2:
        number=weight.iloc[i,2]-0.2
        weight=weight.replace(weight.iloc[i,2], weight.iloc[i,2]-number)
        weight.iloc[i,14]+=number
    if weight.iloc[i,3]>0.2:
        number=weight.iloc[i,3]-0.2
        weight=weight.replace(weight.iloc[i,3], weight.iloc[i,3]-number)
        weight.iloc[i,14]+=number
    if weight.iloc[i,4]>0.2:
        number=weight.iloc[i,4]-0.2
        weight=weight.replace(weight.iloc[i,4], weight.iloc[i,4]-number)
        weight.iloc[i,14]+=number
    if weight.iloc[i,5]>0.2:
        number=weight.iloc[i,5]-0.2
        weight=weight.replace(weight.iloc[i,5], weight.iloc[i,5]-number)
        weight.iloc[i,14]+=number
        
    
        

admomvwpf=pd.DataFrame()
admomvwpf['ret']=mompf['ret']
admomvwpf['ret']=(weight.shift(1).fillna(0).values*ret.fillna(0).values).sum(axis=1)
admomvwpf['logret']=np.log(admomvwpf['ret']+1)
admomvwpf['cumret']=(admomvwpf['ret']+1).cumprod()-1
admomvwpf['logcumret']=admomvwpf['logret'].cumsum()

admomvwpf['logcumret'].plot()

admomvwpf['price']=ret.iloc[:,0].fillna(1)
admomvwpf['price'][0]=1
for i in range(len(momvwpf)-1):
    admomvwpf['price'][i+1]=admomvwpf['price'].values[i]*(1+admomvwpf['ret'].values[i+1])
    
    
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
    cov=(((mom.iloc[i:i+1]>=mom.mean(axis=1)[i]).any()==True)*price[i-5:i+1]).replace(0,np.nan).dropna(axis=1).cov()
    weight_buff=minvol(cov,0,1)
    ret_buff=(((mom.iloc[i:i+1]>=mom.mean(axis=1)[i]).any()==True)*ret[i+1:i+2].replace(0,0.000000001)).replace(0,np.nan).dropna(axis=1)
    minvar['ret'][i+1:i+2]=(ret_buff*weight_buff).sum(axis=1)
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
plt.plot(eqpf['eqcumret']*100, label='EQ Cumret')
plt.plot(vwpf['vwcumret']*100, label='VW Cumret')
plt.plot(mompf['cumret']*100, label='MoM Cumret')
plt.plot(momvwpf['cumret']*100, label='MoMVW Cumret')
plt.plot(minvar['cumret']*100, label='MinVar Cumret')
plt.plot(admomvwpf['cumret']*100, label='ADMoMVW Cumret')
plt.legend(loc='upper left')
plt.show()
    
    
#%% 가격 비교하기

plt.plot(eqpf['price'], label='EQ Cumret')
plt.plot(vwpf['price'], label='VW Cumret')
plt.plot(mompf['price'], label='MoM Cumret')
plt.plot(momvwpf['price'], label='MoMVW Cumret')
plt.plot(minvar['price'], label='MinVar Cumret')
plt.legend(loc='upper left')
plt.show()
    
    
    
    
    
    
    
    
    
    
    
