# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:32:14 2018

@author: 우람
"""

import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt

sp=pd.read_excel('mm.xlsx', index_col='Date', skiprows=[1])
tbill=pd.read_excel('nn.xlsx', index_col='date', skip_footer=1)
tbill.index=sp.index
tbill['tbill']=tbill['tbill']/1200
sp['ret']=sp['price']/sp['price'].shift(-1)-1
logsp=pd.DataFrame(np.log(sp['ret']+1))
logsp['price']=sp['price']
logsp['cumret']=logsp['ret'][::-1].cumsum()
sp['cumret']=(sp['ret']+1)[::-1].cumprod()-1

#%%
sma=pd.rolling_mean(sp['price'][::-1],10)[::-1]

#%%

sp['timingret']=sp['ret']
for i in range(len(sp)):
    if sp['price'][1011-i]<sma[1011-i]:
        sp['timingret'][1010-i]=tbill['tbill'][1011-i]
        
logsp['timingret']=logsp['ret']
for i in range(len(sp)):
    if sp['price'][1011-i]<sma[1011-i]:
        logsp['timingret'][1010-i]=tbill['tbill'][1011-i]
        
#%%
tret=pd.read_excel('hi.xlsx', index_col='Date', skiprows=[1])
tret['ret']=tret['price']/tret['price'].shift(-1)-1
tret['cumret']=(tret['ret']+1)[::-1].cumprod()-1
tret['timingret']=tret['ret']
tsma=pd.rolling_mean(tret['price'][::-1],10)[::-1]
for i in range(len(tret)):
    if tret['price'][363-i]<tsma[363-i]:
        tret['timingret'][362-i]=tbill['tbill'][363-i]

sp['timingcumret']=(sp['timingret']+1)[::-1].cumprod()-1
logsp['timingcumret']=logsp['timingret'][::-1].cumsum()
tret['timingcumret']=(tret['timingret']+1)[::-1].cumprod()-1

#%%  90년부터 12년까지의 누적 수익률 비교하기
#plt.plot(tret['cumret'].iloc[64:340][::-1], label='Total Cumret')
#plt.plot(tret['timingcumret'].iloc[64:340][::-1], label='Total Timing Cumret')
#plt.legend(loc='upper left')
#plt.show()

#%% 최근까지의 누적 수익률 비교하기
#plt.plot(tret['cumret'].iloc[:340][::-1], label='Total Cumret')
#plt.plot(tret['timingcumret'].iloc[:340][::-1], label='Total Timing Cumret')
#plt.legend(loc='upper left')
#plt.show()

#%%

tret['timingprice']=tret['price']
for i in range(len(tret)):
    tret['timingprice'][362-i]=tret['timingprice'][363-i]*(1+tret['timingret'][362-i])
    

#plt.plot(tret['timingprice'].iloc[64:340][::-1], label='Total Price')
#plt.plot(tret['price'].iloc[64:340][::-1], label='Total Price')
#plt.legend(loc='upper left')
#plt.show()
    
print('Total Return의 Sharpe:', (tret['ret'].mean()*12-0.02)/(tret['ret'].std()*12**(1/2)))
print('Total Timing Return의 Sharpe:', (tret['timingret'].mean()*12-0.02)/(tret['timingret'].std()*12**(1/2)))
    

#%% 수익률 히스토그램 그리기
bins=np.linspace(-0.1,0.1,11)
plt.hist([tret['ret'],tret['timingret']],bins, range=[-0.01,0.01])
plt.legend(['Total Ret','Timing Ret'],loc='upper right')
plt.show()

#%% MDD 계산
def MDD(d):
    x=np.maximum.accumulate(d)
    z=((x-d)/x).max(axis=0)
    return z
print('Total Return의 MDD:', MDD(tret['price'][::-1]))
print('Total Timing Return의 MDD:', MDD(tret['timingprice'][-2::-1]))

#pd.rolling_apply(tret['price'][::-1], window=12, func=MDD).plot(subplots=True)  #그래프그리기
#pd.rolling_apply(tret['timingprice'][-2::-1], window=12, func=MDD).plot(subplots=True)
a=pd.rolling_apply(tret['price'][-2::-1], window=12, func=MDD)
b=pd.rolling_apply(tret['timingprice'][-2::-1], window=12, func=MDD)
c= pd.concat([a,b],axis=1)
c.plot()
