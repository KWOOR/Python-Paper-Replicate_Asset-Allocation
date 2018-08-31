# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 23:48:53 2018

@author: 우람
"""

import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt
import scipy as sp
from sklearn import datasets, linear_model
import statsmodels.api as sm
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\nonmun\\GPA')

price=pd.read_excel('close.xlsx').fillna(method='ffill')
pbr=pd.read_excel('pbr.xlsx').fillna(method='ffill')    #낮을수록 좋은 기업!
gp=pd.read_excel('gp.xlsx')
a=pd.read_excel('a.xlsx')
cap=pd.read_excel('cap.xlsx')
MKT=pd.read_excel('MKT.xlsx')
ret=price/price.shift(1)-1
MKTret=MKT/MKT.shift(1)-1
MKTret['cumret']=(MKTret['MKT']+1).cumprod()-1
MKTret=MKTret.fillna(0)
gpa=(gp/a).fillna(method='ffill').fillna(method='bfill')    #높을수록 좋은 기업!

pbr_buff=pbr.rank(axis=1, method='dense')    #낮을수록 좋은 기업! 좋은 기업에 랭크1이 간다!
gpa_buff=(-gpa).rank(axis=1, method='dense')   #(-)를 붙였으니 낮을수록 좋은 기업!! 좋은 기업에 랭크 1이 간다!!
score=pbr_buff+gpa_buff
score_buff=score.rank(axis=1, method='dense')   #낮을수록 좋은거다!!  #스크리닝은 하지 않고, PBR과 GPA를 동시에 고려

#%% 좋은 포트폴리오 만들기
good_score=score_buff.copy()
#for i in range(len(score_buff)):
#    good_score[i:i+1]=(score_buff[i:i+1]<score_buff[i:i+1].count(axis=1).values*0.3)*1 #상위 30% 종목 고르기

for i in range(len(score_buff)):
    good_score[i:i+1]=(score_buff[i:i+1]<score_buff[i:i+1].max(axis=1).values*0.3)*1 #상위 30%를 최대값 대비로 고르기

    
good_ret=good_score*ret
good_weight=good_score*cap
for i in range(len(good_weight)):
    good_weight[i:i+1]=good_weight[i:i+1]/float(good_weight[i:i+1].fillna(0).sum(axis=1))

goodpf=pd.DataFrame()
goodpf['ret']=(good_ret.fillna(0)*good_weight.fillna(0).shift(1)).sum(axis=1)  #빈 값을 0으로 채움
goodpf['logret']=np.log(goodpf['ret']+1)
goodpf['cumret']=(goodpf['ret']+1).cumprod()-1
goodpf['logcumret']=goodpf['logret'].cumsum()

goodpf['cumret'].plot()
#%% 나쁜 포트폴리오 만들기

bad_score=score_buff.copy()
#for i in range(len(score_buff)):
#    bad_score[i:i+1]=(score_buff[i:i+1]>score_buff[i:i+1].count(axis=1).values*0.7)*1 #하위 30% 종목 고르기
    
for i in range(len(score_buff)):
    bad_score[i:i+1]=(score_buff[i:i+1]>score_buff[i:i+1].max(axis=1).values*0.7)*1    #하위 30%를 최대값 대비로 고르기    
    
bad_ret=bad_score*ret
bad_weight=bad_score*cap
for i in range(len(bad_weight)):
    bad_weight[i:i+1]=bad_weight[i:i+1]/float(bad_weight[i:i+1].fillna(0).sum(axis=1))

badpf=pd.DataFrame()
badpf['ret']=(bad_ret.fillna(0)*bad_weight.fillna(0).shift(1)).sum(axis=1) #빈 값을 0으로 채움
badpf['logret']=np.log(badpf['ret']+1)
badpf['cumret']=(badpf['ret']+1).cumprod()-1
badpf['logcumret']=badpf['logret'].cumsum()

badpf['cumret'].plot()

#%% 스크리닝 하기

scr_good=score.copy()
for i in range(len(pbr_buff)):
    scr_good[i:i+1]=(pbr_buff[i:i+1]<pbr_buff[i:i+1].max(axis=1).values*0.5)*1 
scr_good=scr_good*(-gpa)
scr_good=scr_good.replace(-0,np.nan)
scr_good=scr_good.rank(axis=1, method='dense')
for i in range(len(scr_good)):
    scr_good[i:i+1]=(scr_good[i:i+1]<scr_good[i:i+1].max(axis=1).values*0.5)*1 

   
scr_good_ret=scr_good*ret
scr_good_weight=scr_good*cap
for i in range(len(scr_good_weight)):
    scr_good_weight[i:i+1]=scr_good_weight[i:i+1]/float(scr_good_weight[i:i+1].fillna(0).sum(axis=1))

scr_good_pf=pd.DataFrame()
scr_good_pf['ret']=(scr_good_ret.fillna(0)*scr_good_weight.fillna(0).shift(1)).sum(axis=1) #빈 값을 0으로 채움
scr_good_pf['logret']=np.log(scr_good_pf['ret']+1)
scr_good_pf['cumret']=(scr_good_pf['ret']+1).cumprod()-1
scr_good_pf['logcumret']=scr_good_pf['logret'].cumsum()

scr_good_pf['cumret'].plot()


scr_bad=score.copy()
for i in range(len(pbr_buff)):
    scr_bad[i:i+1]=(pbr_buff[i:i+1]>pbr_buff[i:i+1].max(axis=1).values*0.5)*1 
scr_bad=scr_bad*(-gpa)
scr_bad=scr_bad.replace(-0,np.nan)
scr_bad=scr_bad.rank(axis=1, method='dense')
for i in range(len(scr_bad)):
    scr_bad[i:i+1]=(scr_bad[i:i+1]>scr_bad[i:i+1].max(axis=1).values*0.5)*1 

   
scr_bad_ret=scr_bad*ret
scr_bad_weight=scr_bad*cap
for i in range(len(scr_bad_weight)):
    scr_bad_weight[i:i+1]=scr_bad_weight[i:i+1]/float(scr_bad_weight[i:i+1].fillna(0).sum(axis=1))

scr_bad_pf=pd.DataFrame()
scr_bad_pf['ret']=(scr_bad_ret.fillna(0)*scr_bad_weight.fillna(0).shift(1)).sum(axis=1) #빈 값을 0으로 채움
scr_bad_pf['logret']=np.log(scr_bad_pf['ret']+1)
scr_bad_pf['cumret']=(scr_bad_pf['ret']+1).cumprod()-1
scr_bad_pf['logcumret']=scr_bad_pf['logret'].cumsum()

scr_bad_pf['cumret'].plot()

#%%  회귀식
#
#model1=linear_model.LinearRegression()
#x_vars1=['cumret']
#goodtoMKT=model1.fit(goodpf[x_vars1], MKTret['cumret'])
#print( model1.coef_, model1.intercept_)


regressor=goodpf['ret']
regressor=sm.add_constant(regressor)
model1=sm.OLS(MKTret['MKT'], regressor ).fit()
model1.summary()



regressor=badpf['ret']
regressor=sm.add_constant(regressor)
model2=sm.OLS(MKTret['MKT'], regressor ).fit()
model2.summary()


regressor=scr_good_pf['ret']
regressor=sm.add_constant(regressor)
model3=sm.OLS(MKTret['MKT'], regressor ).fit()
model3.summary()


regressor=scr_bad_pf['ret']
regressor=sm.add_constant(regressor)
model4=sm.OLS(MKTret['MKT'], regressor ).fit()
model4.summary()
