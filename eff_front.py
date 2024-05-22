#calculate efficient frontier for a portfolio

import numpy as np
import datetime as dt
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
#define stock list and get summary statistics

stocklist=['CBA','BHP','TLS']
stocks=[stock+'.AX' for stock in stocklist]
endDate=dt.datetime.now()
startDate= endDate-dt.timedelta(days=365)
meanReturns , covMatrix = getData(stocks,startDate,endDate)

#define a weighting vector
weights = np.array([0.3, 0.4 , 0.3])

#calculate performance of this portfolio

returns, std = portfolioPerformance(weights,meanReturns, covMatrix)
#print ( round(100*returns,2), round(100*std,2))

#calculate max SR portfolio
result= maxSR(meanReturns, covMatrix)
print(result)



    