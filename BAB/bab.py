# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

data_max = pd.read_excel('beta_data.xlsx', skiprows=13, index_col='Symbol Name')
stock = pd.read_excel('stock_data.xlsx', skiprows=13, index_col='Symbol Name')
sto = stock.T

data_max_t = data_max.T
df = pd.DataFrame()
b = []
for i in list(data_max_t.columns.values):
    data_max_t_a = data_max_t.sort_values([i], ascending=[True])
    b.append(data_max_t_a.index.values)

f_buffer = []
s_buffer = []
t_buffer = []
four_buffer = []
five_buffer = []
six_buffer = []
seven_buffer = []
eight_buffer = []
nine_buffer = []
ten_buffer = []

for i in range(len(b)):
    # Create 10 Portfolio
    first = b[i][:25]
    second = b[i][25:50]
    third = b[i][50:75]
    four = b[i][75:100]
    five = b[i][100:125]
    six = b[i][125:150]
    seven = b[i][150:175]
    eight = b[i][175:200]
    nine = b[i][200:225]
    ten = b[i][225:]
    
    f_buffer.append(first)
    s_buffer.append(second)
    t_buffer.append(third)
    four_buffer.append(four)
    five_buffer.append(five)
    six_buffer.append(six)
    seven_buffer.append(seven)
    eight_buffer.append(eight)
    nine_buffer.append(nine)
    ten_buffer.append(ten)

f_r = []
for j in range(len(list(sto.columns.values)) - 1):
    f_buffer_r = []
    for i in range(len(f_buffer[0])):
        a = sto[sto.index.str.startswith(f_buffer[j][i])].iloc[0]
        f_buffer_r.append(a[j + 1])
    f_r.append(f_buffer_r)

s_r = []
for j in range(len(list(sto.columns.values)) - 1):
    s_buffer_r = []
    for i in range(len(s_buffer[0])):
        a = sto[sto.index.str.startswith(s_buffer[j][i])].iloc[0]
        s_buffer_r.append(a[j + 1])
    s_r.append(s_buffer_r)

t_r = []
for j in range(len(list(sto.columns.values)) - 1):
    t_buffer_r = []
    for i in range(len(t_buffer[0])):
        a = sto[sto.index.str.startswith(t_buffer[j][i])].iloc[0]
        t_buffer_r.append(a[j + 1])
    t_r.append(t_buffer_r)

four_r = []
for j in range(len(list(sto.columns.values)) - 1):
    four_buffer_r = []
    for i in range(len(four_buffer[0])):
        a = sto[sto.index.str.startswith(four_buffer[j][i])].iloc[0]
        four_buffer_r.append(a[j + 1])
    four_r.append(four_buffer_r)

five_r = []
for j in range(len(list(sto.columns.values)) - 1):
    five_buffer_r = []
    for i in range(len(five_buffer[0])):
        a = sto[sto.index.str.startswith(five_buffer[j][i])].iloc[0]
        five_buffer_r.append(a[j + 1])
    five_r.append(five_buffer_r)

six_r = []
for j in range(len(list(sto.columns.values)) - 1):
    six_buffer_r = []
    for i in range(len(six_buffer[0])):
        a = sto[sto.index.str.startswith(six_buffer[j][i])].iloc[0]
        six_buffer_r.append(a[j + 1])
    six_r.append(six_buffer_r)

seven_r = []
for j in range(len(list(sto.columns.values)) - 1):
    seven_buffer_r = []
    for i in range(len(seven_buffer[0])):
        a = sto[sto.index.str.startswith(seven_buffer[j][i])].iloc[0]
        seven_buffer_r.append(a[j + 1])
    seven_r.append(seven_buffer_r)

eight_r = []
for j in range(len(list(sto.columns.values)) - 1):
    eight_buffer_r = []
    for i in range(len(eight_buffer[0])):
        a = sto[sto.index.str.startswith(eight_buffer[j][i])].iloc[0]
        eight_buffer_r.append(a[j + 1])
    eight_r.append(eight_buffer_r)

nine_r = []
for j in range(len(list(sto.columns.values)) - 1):
    nine_buffer_r = []
    for i in range(len(nine_buffer[0])):
        a = sto[sto.index.str.startswith(nine_buffer[j][i])].iloc[0]
        nine_buffer_r.append(a[j + 1])
    nine_r.append(nine_buffer_r)

ten_r = []
for j in range(len(list(sto.columns.values)) - 1):
    ten_buffer_r = []
    for i in range(len(ten_buffer[0])):
        a = sto[sto.index.str.startswith(ten_buffer[j][i])].iloc[0]
        ten_buffer_r.append(a[j + 1])
    ten_r.append(ten_buffer_r)

f_mean = []
for i in f_r:
    listSum = sum(i)
    listLength = len(i)
    f_mean.append(listSum / listLength)

s_mean = []
for i in s_r:
    listSum = sum(i)
    listLength = len(i)
    s_mean.append(listSum / listLength)

t_mean = []
for i in t_r:
    listSum = sum(i)
    listLength = len(i)
    t_mean.append(listSum / listLength)

four_mean = []
for i in four_r:
    listSum = sum(i)
    listLength = len(i)
    four_mean.append(listSum / listLength)

five_mean = []
for i in five_r:
    listSum = sum(i)
    listLength = len(i)
    five_mean.append(listSum / listLength)

six_mean = []
for i in six_r:
    listSum = sum(i)
    listLength = len(i)
    six_mean.append(listSum / listLength)

seven_mean = []
for i in seven_r:
    listSum = sum(i)
    listLength = len(i)
    seven_mean.append(listSum / listLength)

eight_mean = []
for i in eight_r:
    listSum = sum(i)
    listLength = len(i)
    eight_mean.append(listSum / listLength)

nine_mean = []
for i in nine_r:
    listSum = sum(i)
    listLength = len(i)
    nine_mean.append(listSum / listLength)

ten_mean = []
for i in ten_r:
    listSum = sum(i)
    listLength = len(i)
    ten_mean.append(listSum / listLength)

new = pd.concat([pd.DataFrame(f_mean), pd.DataFrame(s_mean), pd.DataFrame(t_mean),
                 pd.DataFrame(four_mean), pd.DataFrame(five_mean), pd.DataFrame(six_mean),
                 pd.DataFrame(seven_mean), pd.DataFrame(eight_mean) ,pd.DataFrame(nine_mean),
                 pd.DataFrame(ten_mean)], axis=1)
new.to_excel("bab_result.xlsx")
