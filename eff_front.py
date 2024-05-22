#calculate efficient frontier for a portfolio

import numpy as np
import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import scipy.optimize as sc
yf.pdr_override()

# get data

def getData(stocks,start,end):
    stockData=pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData=stockData['Close']
    
    returns=stockData.pct_change()
    meanReturns=returns.mean()
    covMatrix=returns.cov()
    return meanReturns,covMatrix

# define function to calculate performance of portfolio

def portfolioPerformance(weights, meanReturns,covMatrix):
    returns=np.sum(weights * meanReturns)*252 #go from daily to yearly
    std=np.sqrt(np.dot(weights.T, np.dot(covMatrix , weights) ) ) * np.sqrt(252)
    return returns , std

def negSR(weights, meanReturns, covMatrix,riskFreeRate=0):
    pReturns, pStd = portfolioPerformance(weights,meanReturns, covMatrix)
    return (riskFreeRate-pReturns)/pStd

def maxSR(meanReturns, covMatrix,riskFreeRate=0,constraintSet=(0,1)):
    #minimize the negative SR by altering weights
    numAssets=len(meanReturns)
    args=(meanReturns,covMatrix,riskFreeRate)
    constraints= ( {'type':'eq' , 'fun' : lambda x: np.sum(x) -1})
    bound=constraintSet
    bounds=tuple(bound for asset in range (numAssets))
    result = sc.minimize(negSR, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioVariance(weights, meanReturns,covMatrix):
    return portfolioPerformance(weights, meanReturns,covMatrix)[1]

def minVar(meanReturns, covMatrix,riskFreeRate=0,constraintSet=(0,1)):
    #minimize variance by altering weights
    numAssets=len(meanReturns)
    args=(meanReturns,covMatrix)
    constraints= ( {'type':'eq' , 'fun' : lambda x: np.sum(x) -1})
    bound=constraintSet
    bounds=tuple(bound for asset in range (numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result
    
#define stock list and get summary statistics
stocklist=['CBA','BHP','TLS']
stocks=[stock+'.AX' for stock in stocklist]
endDate=dt.datetime.now()
startDate= endDate-dt.timedelta(days=365)
meanReturns , covMatrix = getData(stocks,startDate,endDate)



def calculatedResults(meanReturns, covMatrix,riskFreeRate=0,constraintSet=(0,1)):
    #Output Max SR, Min Volatility, efficient frontier
    maxSR_Portfolio= maxSR(meanReturns, covMatrix)
    maxSR_Returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns,covMatrix)
    maxSR_Returns, maxSR_std= round(100*maxSR_Returns,2), round(100*maxSR_std,2)
    maxSR_allocation=pd.DataFrame(maxSR_Portfolio['x'],index=meanReturns.index,columns=['allocation'])
    maxSR_allocation.allocation=[round(i*100,1) for i in maxSR_allocation.allocation]
    
    minVar_Portfolio= minVar(meanReturns, covMatrix)
    minVar_Returns, minVar_std = portfolioPerformance(minVar_Portfolio['x'], meanReturns,covMatrix)
    minVar_Returns, minVar_std= round(100*minVar_Returns,2), round(100*minVar_std,2)
    minVar_allocation=pd.DataFrame(minVar_Portfolio['x'],index=meanReturns.index,columns=['allocation'])
    minVar_allocation.allocation=[round(i*100,1) for i in minVar_allocation.allocation]
    return maxSR_Returns, maxSR_std, maxSR_allocation ,minVar_Returns, minVar_std, minVar_allocation

result= calculatedResults(meanReturns, covMatrix,riskFreeRate=0,constraintSet=(0,1))
a,b= result[2], result[5]
print(a+b)