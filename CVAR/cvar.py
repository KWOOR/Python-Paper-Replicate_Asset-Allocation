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
#y_rep=y.copy()
exp_ret=y/y.shift(1)-1
exp_ret=exp_ret.replace(np.inf, 0)
exp_ret=(exp_ret+1).resample('M').prod()-1  #월별 수익률로...

#%% 기대수익률 행렬을 input data로 쓸 건데, sacle이 안 맞으므로 이를 고치기 위해 정규화 시킴

exp_ret_buff=exp_ret.copy()
for i in range(len(exp_ret)):
    exp_ret_buff.iloc[i:i+1]=(exp_ret_buff.iloc[i:i+1].values-exp_ret_buff.iloc[i:i+1].min(axis=1).values)/(exp_ret_buff.iloc[i:i+1].max(axis=1).values-exp_ret_buff.iloc[i:i+1].min(axis=1).values)

#%% 99% CVAR PF 순차적으로 올라가기.. 

def constraints(x):
    return 1- float(pd.DataFrame(x).sum().values)
    
def CVAR(x):    # 99% CVAR 가정
    a=pd.DataFrame(x).T
    a.columns=exp_ret.columns
    b=pd.concat([a,a,a,a,a,a], axis=0)
    vol=(monthly_ret.iloc[i+18:i+24]*b.values).sum(axis=1).std()       # 6개월치 표준편차 계산
   # VaR=(exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-2.33*vol
    CVaR=((exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-(vol/(0.01*np.sqrt(2*np.pi)))*np.exp(-(2.33**2)/2)).values
    #VaR=exp_ret.iloc[i:i+1]*x[0]
    return float(0.01+CVaR)   # CVaR이 0.01보다 작아야하는 제약식

def CVAR2(x):    # 99% CVAR 가정
    a=pd.DataFrame(x).T
    a.columns=exp_ret.columns
    b=pd.concat([a,a,a,a,a,a], axis=0)
    vol=(monthly_ret.iloc[i+18:i+24]*b.values).sum(axis=1).std()       # 6개월치 표준편차 계산
   # VaR=(exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-2.33*vol
    CVaR=((exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-(vol/(0.01*np.sqrt(2*np.pi)))*np.exp(-(2.33**2)/2)).values
    #VaR=exp_ret.iloc[i:i+1]*x[0]
    return float(0.05+CVaR)   # CVaR이 0.05보다 작아야하는 제약식

def CVAR3(x):    # 99% CVAR 가정
    a=pd.DataFrame(x).T
    a.columns=exp_ret.columns
    b=pd.concat([a,a,a,a,a,a], axis=0)
    vol=(monthly_ret.iloc[i+18:i+24]*b.values).sum(axis=1).std()       # 6개월치 표준편차 계산
   # VaR=(exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-2.33*vol
    CVaR=((exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-(vol/(0.01*np.sqrt(2*np.pi)))*np.exp(-(2.33**2)/2)).values
    #VaR=exp_ret.iloc[i:i+1]*x[0]
    return float(0.1+CVaR)   # CVaR이 0.1보다 작아야하는 제약식

def CVAR4(x):    # 99% CVAR 가정
    a=pd.DataFrame(x).T
    a.columns=exp_ret.columns
    b=pd.concat([a,a,a,a,a,a], axis=0)
    vol=(monthly_ret.iloc[i+18:i+24]*b.values).sum(axis=1).std()       # 6개월치 표준편차 계산
   # VaR=(exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-2.33*vol
    CVaR=((exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-(vol/(0.01*np.sqrt(2*np.pi)))*np.exp(-(2.33**2)/2)).values
    #VaR=exp_ret.iloc[i:i+1]*x[0]
    return float(0.2+CVaR)   # CVaR이 0.2보다 작아야하는 제약식

def RETURN(x):
    return -float((exp_ret[i:i+1]*x.T).sum(axis=1).values)#-(exp_ret[i:i+1].T.dropna().T*x).sum(axis=1)

weight=list()
for i in range(len(exp_ret)):
    #weight_buff=np.repeat(1/exp_ret.iloc[i:i+1].notnull().sum(axis=1).values,exp_ret.iloc[i:i+1].notnull().sum(axis=1).values)
    #lbound=np.repeat(0, exp_ret.iloc[i:i+1].notnull().sum(axis=1))
    #ubound=np.repeat(1, exp_ret.iloc[i:i+1].notnull().sum(axis=1))
    weight_buff=np.array(exp_ret_buff.iloc[i:i+1]).T    #input data로 앞에서 만든 정규화한 행렬 사용
    lbound=np.repeat(0,len(exp_ret_buff.T))
    ubound=np.repeat(1,len(exp_ret_buff.T))
    bnds=tuple(zip(lbound, ubound))
    const=({'type':'eq', 'fun': constraints}, {'type':'ineq', 'fun':CVAR})
    result=sp.optimize.minimize(RETURN,weight_buff,method='SLSQP', constraints=const, bounds=bnds)   #Nelder-Mead 방식으로하면, 제약식이 안 먹히긴 하지만 대략 1 비슷하게 나오고, Nan값 없음!
    if result.success==True:
        weight.append(result.x)
    else:
        const=({'type':'eq', 'fun': constraints}, {'type':'ineq', 'fun':CVAR2})
        result=sp.optimize.minimize(RETURN, weight_buff,method='SLSQP', constraints=const, bounds=bnds)
        if result.success ==True:
            weight.append(result.x)
        else:
            const=({'type':'eq', 'fun': constraints}, {'type':'ineq', 'fun':CVAR3})
            result=sp.optimize.minimize(RETURN, weight_buff, method='SLSQP', constraints=const, bounds=bnds)
            if result.success ==True:
                weight.append(result.x)
            else:
                const=({'type':'eq', 'fun': constraints}, {'type':'ineq', 'fun':CVAR4})
                result=sp.optimize.minimize(RETURN, weight_buff, method='SLSQP', constraints=const, bounds=bnds)
                if result.success ==True:
                    weight.append(result.x)
                else:
                    const=({'type':'eq', 'fun': constraints})
                    result=sp.optimize.minimize(RETURN, weight_buff, method='SLSQP', constraints=const, bounds=bnds)
                    weight.append(result.x)
                    print(result.success)

CVaR_weight1=pd.DataFrame(weight, columns=exp_ret.columns,index=exp_ret.index)

CVaR1=pd.DataFrame()
CVaR1['ret']=monthly_ret[23:].sum(axis=1)
CVaR1['ret']=(CVaR_weight1.shift(1).fillna(0).values*monthly_ret[23:].fillna(0).values).sum(axis=1)
CVaR1['logret']=np.log(CVaR1['ret']+1)
CVaR1['cumret']=(CVaR1['ret']+1).cumprod()-1
CVaR1['logcumret']=CVaR1['logret'].cumsum()

CVaR1['logcumret'].plot()

CVaR1['price']=monthly_ret.iloc[:,0].fillna(1)
CVaR1['price'][0]=1
for i in range(len(CVaR1)-1):
    CVaR1['price'][i+1]=CVaR1['price'].values[i]*(1+CVaR1['ret'].values[i+1])


#%%  95% CVAR PF 순차적으로 올리기

def constraints(x):
    return 1- float(pd.DataFrame(x).sum().values)
    
def CVAR(x):    # 95% CVAR 가정
    a=pd.DataFrame(x).T
    a.columns=exp_ret.columns
    b=pd.concat([a,a,a,a,a,a], axis=0)
    vol=(monthly_ret.iloc[i+18:i+24]*b.values).sum(axis=1).std()       # 6개월치 표준편차 계산
   # VaR=(exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-2.33*vol
    CVaR=((exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-(vol/(0.05*np.sqrt(2*np.pi)))*np.exp(-(1.65**2)/2)).values
    #VaR=exp_ret.iloc[i:i+1]*x[0]
    return float(0.01+CVaR)   # CVaR이 0.01보다 작아야하는 제약식

def CVAR2(x):    # 95% CVAR 가정
    a=pd.DataFrame(x).T
    a.columns=exp_ret.columns
    b=pd.concat([a,a,a,a,a,a], axis=0)
    vol=(monthly_ret.iloc[i+18:i+24]*b.values).sum(axis=1).std()       # 6개월치 표준편차 계산
   # VaR=(exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-2.33*vol
    CVaR=((exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-(vol/(0.05*np.sqrt(2*np.pi)))*np.exp(-(1.65**2)/2)).values
    #VaR=exp_ret.iloc[i:i+1]*x[0]
    return float(0.05+CVaR)   # CVaR이 0.05보다 작아야하는 제약식

def CVAR3(x):    # 95% CVAR 가정
    a=pd.DataFrame(x).T
    a.columns=exp_ret.columns
    b=pd.concat([a,a,a,a,a,a], axis=0)
    vol=(monthly_ret.iloc[i+18:i+24]*b.values).sum(axis=1).std()       # 6개월치 표준편차 계산
   # VaR=(exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-2.33*vol
    CVaR=((exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-(vol/(0.05*np.sqrt(2*np.pi)))*np.exp(-(1.65**2)/2)).values
    #VaR=exp_ret.iloc[i:i+1]*x[0]
    return float(0.1+CVaR)   # CVaR이 0.1보다 작아야하는 제약식

def CVAR4(x):    # 95% CVAR 가정
    a=pd.DataFrame(x).T
    a.columns=exp_ret.columns
    b=pd.concat([a,a,a,a,a,a], axis=0)
    vol=(monthly_ret.iloc[i+18:i+24]*b.values).sum(axis=1).std()       # 6개월치 표준편차 계산
   # VaR=(exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-2.33*vol
    CVaR=((exp_ret.iloc[i:i+1]*a.values).sum(axis=1)-(vol/(0.05*np.sqrt(2*np.pi)))*np.exp(-(1.65**2)/2)).values
    #VaR=exp_ret.iloc[i:i+1]*x[0]
    return float(0.2+CVaR)   # CVaR이 0.2보다 작아야하는 제약식

def RETURN(x):
    return -float((exp_ret[i:i+1]*x.T).sum(axis=1).values)#-(exp_ret[i:i+1].T.dropna().T*x).sum(axis=1)

weight=list()
for i in range(len(exp_ret)):
    #weight_buff=np.repeat(1/exp_ret.iloc[i:i+1].notnull().sum(axis=1).values,exp_ret.iloc[i:i+1].notnull().sum(axis=1).values)
    #lbound=np.repeat(0, exp_ret.iloc[i:i+1].notnull().sum(axis=1))
    #ubound=np.repeat(1, exp_ret.iloc[i:i+1].notnull().sum(axis=1))
    weight_buff=np.array(exp_ret_buff.iloc[i:i+1]).T   #input 데이터로 앞에서와 같이 정규화한 행렬 사용
    lbound=np.repeat(0,len(exp_ret.T))
    ubound=np.repeat(1,len(exp_ret.T))
    bnds=tuple(zip(lbound, ubound))
    const=({'type':'eq', 'fun': constraints}, {'type':'ineq', 'fun':CVAR})
    result=sp.optimize.minimize(RETURN,weight_buff,method='SLSQP', constraints=const, bounds=bnds)   #Nelder-Mead 방식으로하면, 제약식이 안 먹히긴 하지만 대략 1 비슷하게 나오고, Nan값 없음!
    if result.success==True:
        weight.append(result.x)
    else:
        const=({'type':'eq', 'fun': constraints}, {'type':'ineq', 'fun':CVAR2})
        result=sp.optimize.minimize(RETURN, weight_buff,method='SLSQP', constraints=const, bounds=bnds)
        if result.success ==True:
            weight.append(result.x)
        else:
            const=({'type':'eq', 'fun': constraints}, {'type':'ineq', 'fun':CVAR3})
            result=sp.optimize.minimize(RETURN, weight_buff, method='SLSQP', constraints=const, bounds=bnds)
            if result.success ==True:
                weight.append(result.x)
            else:
                const=({'type':'eq', 'fun': constraints}, {'type':'ineq', 'fun':CVAR4})
                result=sp.optimize.minimize(RETURN, weight_buff, method='SLSQP', constraints=const, bounds=bnds)
                if result.success ==True:
                    weight.append(result.x)
                else:
                    const=({'type':'eq', 'fun': constraints})
                    result=sp.optimize.minimize(RETURN, weight_buff, method='SLSQP', constraints=const, bounds=bnds)
                    weight.append(result.x)
                    print(result.success)

CVaR_weight2=pd.DataFrame(weight, columns=exp_ret.columns,index=exp_ret.index)

CVaR2=pd.DataFrame()
CVaR2['ret']=monthly_ret[23:].sum(axis=1)
CVaR2['ret']=(CVaR_weight2.shift(1).fillna(0).values*monthly_ret[23:].fillna(0).values).sum(axis=1)
CVaR2['logret']=np.log(CVaR2['ret']+1)
CVaR2['cumret']=(CVaR2['ret']+1).cumprod()-1
CVaR2['logcumret']=CVaR2['logret'].cumsum()

CVaR2['logcumret'].plot()

CVaR2['price']=monthly_ret.iloc[:,0].fillna(1)
CVaR2['price'][0]=1
for i in range(len(CVaR2)-1):
    CVaR2['price'][i+1]=CVaR2['price'].values[i]*(1+CVaR2['ret'].values[i+1])



#%%

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



#%%   CVAR 없이, 기대수익률만 극대화하는 법
weight=list()
for i in range(len(exp_ret)):
    #weight_buff=np.repeat(1/exp_ret.iloc[i:i+1].notnull().sum(axis=1).values,exp_ret.iloc[i:i+1].notnull().sum(axis=1).values)
    #lbound=np.repeat(0, exp_ret.iloc[i:i+1].notnull().sum(axis=1))
    #ubound=np.repeat(1, exp_ret.iloc[i:i+1].notnull().sum(axis=1))
    weight_buff=np.array(exp_ret_buff.iloc[i:i+1]).T
    lbound=np.repeat(0,len(exp_ret.T))
    ubound=np.repeat(1,len(exp_ret.T))
    bnds=tuple(zip(lbound, ubound))
    const=({'type':'eq', 'fun': constraints})
    result=sp.optimize.minimize(RETURN,weight_buff,method='SLSQP', constraints=const, bounds=bnds)   #Nelder-Mead 방식으로하면, 제약식이 안 먹히긴 하지만 대략 1 비슷하게 나오고, Nan값 없음!
    weight.append(result.x)

CVaR_weight3=pd.DataFrame(weight, columns=exp_ret.columns,index=exp_ret.index)

CVaR3=pd.DataFrame()
CVaR3['ret']=monthly_ret[23:].sum(axis=1)
CVaR3['ret']=(CVaR_weight3.shift(1).fillna(0).values*monthly_ret[23:].fillna(0).values).sum(axis=1)
CVaR3['logret']=np.log(CVaR3['ret']+1)
CVaR3['cumret']=(CVaR3['ret']+1).cumprod()-1
CVaR3['logcumret']=CVaR3['logret'].cumsum()

CVaR3['logcumret'].plot()

CVaR3['price']=monthly_ret.iloc[:,0].fillna(1)
CVaR3['price'][0]=1
for i in range(len(CVaR3)-1):
    CVaR3['price'][i+1]=CVaR3['price'].values[i]*(1+CVaR3['ret'].values[i+1])


#%%   결과 정리

def MDD(d):
    x=np.maximum.accumulate(d)
    z=((x-d)/x).max(axis=0)
    return z

print('99% CVaR Return의 Sharpe:', (CVaR1['ret'].mean()*12-0.02)/(CVaR1['ret'].std()*(12**(1/2))))  #단순비교를 위한거라서 기하수익률이 아닌 산술평균을 이용했음
print('99% CVaR Return의 MDD:', MDD(CVaR1['price']))
print('95% CVaR2 Return의 Sharpe:', (CVaR2['ret'].mean()*12-0.02)/(CVaR2['ret'].std()*(12**(1/2))))  #단순비교를 위한거라서 기하수익률이 아닌 산술평균을 이용했음
print('95% CVaR2 Return의 MDD:', MDD(CVaR2['price']))
print('No Use CVaR Return의 Sharpe:', (CVaR3['ret'].mean()*12-0.02)/(CVaR3['ret'].std()*(12**(1/2))))  #단순비교를 위한거라서 기하수익률이 아닌 산술평균을 이용했음
print('No Use CVaR Return의 MDD:', MDD(CVaR3['price']))


plt.plot(CVaR1['cumret']*100, label='99% CVaR Cumret')
plt.plot(CVaR2['cumret']*100, label='95% CVaR Cumret')
plt.plot(CVaR3['cumret']*100, label='No Use CVaR Cumret')
plt.legend(loc='upper left')
plt.show()
