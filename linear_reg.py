import numpy as np # for fast vector computations
import pandas as pd # for easy data analysis
import matplotlib.pyplot as plt # for plotting
from sklearn import linear_model # for linear regression

df = pd.read_csv('swapLiborData.csv')
df['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['Date'],'D')
print(df.head())
#Write a function that given two dates d1 <= d2, returns a dataframe with all the Libor rates in that time interval.

def libor_rates_time_window(df, d1, d2):
    ''' Retrieve the Libor rates (all terms) for the date window d1 to d2. '''
    
    sub_df=df[(df['Date'] >= d1) & (df['Date'] <= d2)]
    sub_df=sub_df[['Date','US0001M','US0002M','US0003M','US0006M','US0012M']]
    return sub_df
print(libor_rates_time_window(df, '2017-12-12', '2018-05-25'))

# we fix a time window
d1 = '2014-01-01'
d2 = '2016-05-24'
# we extract the data for the time window
sub_df = libor_rates_time_window(df, d1, d2)

# the rate we want to predict
y_name = ['US0012M']
# the regressor
x_name = ['US0001M']

#use linear regression

xX=sub_df[x_name]
yY=sub_df[y_name]
regr=linear_model.LinearRegression()
regr.fit(xX,yY)
b_1=regr.coef_
b_0=regr.intercept_
R_2=regr.score(xX,yY)
#plot

yPredict = b_0 + xX * b_1
plt.figure(figsize=(8,6))
plt.plot(sub_df.Date, yPredict)
plt.plot(sub_df.Date, yY)
plt.legend(['constructed 12-month Libor', 'real 12-month Libor'])
plt.show()


# the rate we want to predict
y_name = ['US0012M']
# the regressors
x_name = ['US0002M', 'US0003M', 'US0006M']


xX=sub_df[x_name]
yY=sub_df[y_name]
regr=linear_model.LinearRegression()
regr.fit(xX,yY)
b=regr.coef_
b_0=regr.intercept_
R_2=regr.score(xX,yY)
b_1=b[0,0]
b_2=b[0,1]
b_3=b[0,2]
print(b)
print(b_1,b_2,b_3)

y_hat = b_0 + xX.US0002M * b_1 + xX.US0003M * b_2 + xX.US0006M * b_3
plt.figure(figsize=(8,6))
plt.plot(sub_df.Date, y_hat)
plt.plot(sub_df.Date, yY)
plt.legend(['constructed 12-month Libor', 'real 12-month Libor'])
plt.show()
