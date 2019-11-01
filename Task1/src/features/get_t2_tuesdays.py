# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:18:34 2019

@author: FranciscoP.Romero
"""

import os
import pandas as pd
import datetime

#0 . Load the data 
# read the csv
df = pd.read_csv("T2.csv")
# list the columns
list(df)
# print number of rows and columns 
print (df.shape)

# 1. Filtering

# 1.1 Filter rows
# convert string to datetime .... Be careful!!! Spelling errors!!!
df['TimeStemp'] = pd.to_datetime(df['TimeStemp'])
# extract date from datetime
df['date'] = [d.date() for d in df['TimeStemp']]
# list the available days
print(df['date'].unique())
#filter data by date

df.date = pd.to_datetime(df.date) 
df_tuesdays = df[df['date'].isin(pd.bdate_range(
    '2016-04-01',
    '2016-06-01',
    freq=pd.offsets.CustomBusinessDay(weekmask='Tue')))]
print(df_tuesdays.shape)
print(df_tuesdays.head())
print(df_tuesdays['date'].unique())

df_tuesdays.drop(['date', 'TimeStemp'], axis=1, inplace=True)
df_tuesdays.to_csv(os.path.join('.', 'data', 'processed', 'T2_tuesdays.csv'))


'''df2 = df[(df['TimeStemp'] > '2016-04-30 00:00:00') & (df['TimeStemp'] <= '2016-04-30 23:59:59')]
print(df2.head())
print (df2.shape)
df28.to_csv("T2_out.csv")'''