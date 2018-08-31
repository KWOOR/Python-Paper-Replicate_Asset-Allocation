# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:02:22 2018

@author: 우람
"""


import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt
import scipy as sp
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\nonmun\\Enhanced')


price=pd.read_excel('emp_price_20180614.xlsx', index_col='DATE')
ret=pd.DataFrame(columns=price.columns)
ret=price/price.shift(1)-1
ret=(ret+1).resample('M').prod()-1
price=price.resample("M").mean()
logret=np.log(ret+1)

#%% MoM 포트폴리오 만들기 모멘텀 있는 종목들 추려서 EQ로 투자함..
mom=np.exp(logret.rolling(window=6, center=False).sum())-1
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
    
#%% MoM VW    수익률의 표준편차를 곱하면.. 성과가 쓰레기가 됨. 그래서 가격의 표준편차를 곱했음. 근데 수익률의 표준편차를 쓰는게 맞
vol=pd.DataFrame(columns=price.columns)
vol=1/price.rolling(window=3).std()  
  
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
#%% KODEX200 ETF 떼어내기
etf=pd.DataFrame()
etf['ret']=ret.iloc[:,0]
etf['logret']=np.log(etf['ret']+1)
etf['cumret']=(etf['ret']+1).cumprod()-1
etf['logcumret']=etf['logret'].cumsum()

etf['logcumret'].plot()

etf['price']=ret.iloc[:,0].fillna(1)
etf['price'][0]=1
for i in range(len(etf)-1):
    etf['price'][i+1]=etf['price'].values[i]*(1+etf['ret'].values[i+1])

#%% Enhanced 만들기 - 역분산 결합 방식.. 위에선 가격의 표준편차를 썼지만 여기선 수익률의 표준편차를 씀
    #momvwpf에 가격이 없기 때문이기도 하지만.. 가격을 사용할 땐, 스케일이 달라서 무리가 있음.
en_ret=pd.concat([momvwpf['ret'], etf['ret']], axis=1)
en_ret.columns=['momvwpf', 'etf']
etf_weight=pd.DataFrame()
etf_weight=(1/etf['ret'].rolling(window=3).std())/((1/etf['ret'].rolling(window=3).std())+(1/momvwpf['ret'].rolling(window=3).std())) 
momvw_weight=1-etf_weight
en_weight=pd.concat([momvw_weight,etf_weight], axis=1)
en_weight.columns=['momvwpf', 'etf']

en_vw_pf=pd.DataFrame()
en_vw_pf['ret']=(en_ret*en_weight.shift(1)).sum(axis=1)
en_vw_pf['logret']=np.log(en_vw_pf['ret']+1)
en_vw_pf['cumret']=(en_vw_pf['ret']+1).cumprod()-1
en_vw_pf['logcumret']=en_vw_pf['logret'].cumsum()

en_vw_pf['logcumret'].plot()

en_vw_pf['price']=ret.iloc[:,0].fillna(1)
en_vw_pf['price'][0]=1
for i in range(len(en_vw_pf)-1):
    en_vw_pf['price'][i+1]=en_vw_pf['price'].values[i]*(1+en_vw_pf['ret'].values[i+1])

#%% Enhanced 만들기- 정보비율 최대화
def constraints(x):
    return x.sum()-1 

    
weight=list()
for i in range(6,len(en_weight)):
    weight_buff=np.repeat(1/2, 2)
    lbound=np.repeat(0, 2)
    ubound=np.repeat(1, 2)
    bnds=tuple(zip(lbound, ubound))
    const=({'type':'eq', 'fun': constraints})
    def IR(x):
        a=x[0]*momvwpf['ret'].iloc[i:i+1]+x[1]*etf['ret'].iloc[i:i+1]
        b=np.sqrt((x[0]**2)*momvwpf['ret'].iloc[i-2:i+1].var() + (x[1]**2)*etf['ret'].iloc[i-2:i+1].var() + 2*x[0]*x[1]* np.cov(momvwpf['ret'].iloc[i-2:i+1], etf['ret'].iloc[i-2:i+1])[0][1])
        return -a/b
    result=sp.optimize.minimize(IR, weight_buff,method='SLSQP', constraints=const, bounds=bnds)   #Nelder-Mead 방식으로하면, 제약식이 안 먹히긴 하지만 대략 1 비슷하게 나오고, Nan값 없음!
    weight.append(result.x)
    #weight.iloc[i:i+1]=result.x

weight=pd.DataFrame(weight, columns=en_ret.columns,index=ret[6:].index )
buff=pd.DataFrame(columns=en_ret.columns, index=ret[:6].index)
weight=pd.concat([buff,weight])
    

en_ir_pf=pd.DataFrame()
en_ir_pf['ret']=(en_ret*weight.shift(1)).sum(axis=1)
en_ir_pf['logret']=np.log(en_ir_pf['ret']+1)
en_ir_pf['cumret']=(en_ir_pf['ret']+1).cumprod()-1
en_ir_pf['logcumret']=en_ir_pf['logret'].cumsum()

en_ir_pf['logcumret'].plot()

en_ir_pf['price']=ret.iloc[:,0].fillna(1)
en_ir_pf['price'][0]=1
for i in range(len(en_ir_pf)-1):
    en_ir_pf['price'][i+1]=en_ir_pf['price'].values[i]*(1+en_ir_pf['ret'].values[i+1])


    
#%% 누적 수익률 비교하기
plt.plot(momvwpf['cumret']*100, label='MoMVW Cumret')
plt.plot(etf['cumret']*100, label='ETF Cumret')
plt.plot(en_vw_pf['cumret']*100, label='EN_VW_PF Cumret')
plt.plot(en_ir_pf['cumret']*100, label='EN_IR_PF Cumret')
plt.legend(loc='upper left')
plt.show()
    
    
    
    
    
    
    
    
    
    






















    
