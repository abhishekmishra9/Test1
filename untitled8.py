# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:33:52 2017

@author: Abhishek_Mishra9
"""

from sklearn import linear_model  
import numpy as np
import statsmodels.api as sm
import pandas
import datetime
import numpy as np
import pandas as pd
filename = 'C:/Users/abhishek_mishra9/Desktop/DATA-ANALYSIS/orderdata_TEST1170124093145.csv'
df=pandas.read_csv(filename, sep=',', delimiter=None)
df.IRDER_DATE=pandas.to_datetime(df.ORDER_DATE)
df.Delievery_Date=pandas.to_datetime(df.Delievery_Date)
df.EstimatedDeliveryDateMax=pandas.to_datetime(df.EstimatedDeliveryDateMax)
df.Delievery_Date=df.Delievery_Date-df.ORDER_DATE
df.ORDER_DATE=df.EstimatedDeliveryDateMax-df.ORDER_DATE
df.Delievery_Date=df.Delievery_Date/np.timedelta64(1, 'D')
df.ORDER_DATE=df.ORDER_DATE/np.timedelta64(1, 'D')
df=df.drop('EstimatedDeliveryDateMax',1)
df=df.drop('Base_SKUs',1)
k=df.Delievery_Date.tolist()
y=df.ORDER_DATE.tolist()
x=[]
x1=df.PLT.tolist()
x2=df.PLTHoliday.tolist();
x3=df.MLT.tolist();
x4=df.MLTHoliday.tolist();
x5=df.CFI.tolist();
x6=df.CFS.tolist();
x7=df.Conus.tolist();
x8=df.Quantity.tolist();
x9=df.PlannedEvent.tolist();
x10=df.LargeItem.tolist();
x11=df.LLT.tolist();
x12=df.LLTHoliday.tolist();
x13=df.ShuttleLeadDays.tolist();
x14=df.FCTransitLeadDays.tolist();
x.append(x1);
x.append(x2);
x.append(x4);
x.append(x5);
x.append(x6);
x.append(x7);
x.append(x8);
x.append(x9);
x.append(x10);
x.append(x11);
x.append(x12);
x.append(x13);
x.append(x14);


z=[]
z.append(x1);
z.append(x2);
z.append(x4);
z.append(x5);
z.append(x6);
z.append(x7);
z.append(x8);
z.append(x9);
z.append(x10);
z.append(x11);
z.append(x12);
z.append(x13);
z.append(x14);



ones = np.ones(len(x[0]))
X = sm.add_constant(np.column_stack((x[0], ones)))
for ele in x[1:]:
    X = sm.add_constant(np.column_stack((ele, X)))
    model = linear_model.LinearRegression()  
    result=model.fit(X, y)