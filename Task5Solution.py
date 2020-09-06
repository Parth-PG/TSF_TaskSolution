# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 13:37:48 2020

@author: Parth Gupta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('SampleSuperstore.csv')
print(df.head(5))
print(df.ndim)
print(df.shape)
print(df.describe())

df['Ship Mode'].value_counts()

import seaborn as sns
sns.countplot(x='Ship Mode', data=df )

df['City'].value_counts().plot(kind = 'bar')


df['Region'].value_counts()
sns.countplot(x = 'Region', data = df )

df['Sub-Category'].value_counts().plot(kind = 'bar')

sns.distplot(df['Sales'] )
sns.distplot(df['Quantity'] )
sns.distplot(df['Profit'] )

df['State'].value_counts().head(10).plot(kind = 'bar')

plt.figure(figsize = (9, 9))
sns.stripplot(x='Category',y='Profit',data=df ,hue='Ship Mode')

temp1 = df.groupby(['Segment'],as_index = False).sum()
sns.barplot(x = 'Segment', y = 'Quantity', data = temp1)

temp2 = df.groupby(['Ship Mode'], as_index = False)['Profit'].sum()
sns.barplot(temp2['Ship Mode'], temp2['Profit'])

data1 = df.groupby(['Ship Mode'])['Discount'].sum().reset_index()
data2 = df.groupby(['Region'])['Discount'].sum().reset_index()
data3 = df.groupby(['Region'], as_index = False, sort = True)['Profit'].mean()
data4 = df.groupby('Category')['Profit','Quantity'].sum()


labels = data1['Ship Mode'].values
sizes = data1['Discount'].values
fig1, axis = plt.subplots(figsize=(8,8))
axis.pie(sizes, labels=labels, autopct='%1.2f%%',
        startangle=90)
plt.title('Discount in different Ship Mode', fontsize=22)
plt.show()

labels = data2['Region'].values
sizes = data2['Discount'].values
fig1, ax1 = plt.subplots(figsize=(7,7))
ax1.pie(sizes, labels=labels, autopct='%1.2f%%',
         startangle=90)
plt.title('Discount in different Region', fontsize=20)
plt.show()

region = data3['Region'].values
profit = data3['Profit'].values
fig1, ax1 = plt.subplots(figsize=(7,7))
ax1.pie(profit, labels = region, autopct='%1.2f%%')
plt.title('Average profit for Different regions', fontsize = 20)
plt.figure(figsize = (7, 7))
df['Sub-Category'].value_counts().plot(kind = 'pie', autopct='%1.1f%%' )
plt.show()


data4.plot( kind ="bar" ,  color = ['red','blue'] , figsize= (7,4))
plt.ylabel("Total Profits and Quantities ")
plt.title(" Profit vs Quantity  in Category ")
plt.legend()
plt.show()


corr = df.corr()
sns.heatmap(corr, vmax=.3, center=0, linewidths=.5, cbar_kws={"shrink": .5} , annot = True)













