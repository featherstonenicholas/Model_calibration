#calculate efficient frontier for a portfolio

import numpy as np
import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import scipy.optimize as sc
import plotly.graph_objects as go
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
def portfolioReturn(weights, meanReturns,covMatrix):
    return portfolioPerformance(weights, meanReturns,covMatrix)[0]
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

def efficientOpt(meanReturns, covMatrix,returnTarget,riskFreeRate=0,constraintSet=(0,1)):
    #return min variance portfolio for target return
    numAssets=len(meanReturns)
    args=(meanReturns,covMatrix)
         
    constraints= ( {'type':'eq' , 'fun' : lambda x: portfolioReturn(x,meanReturns,covMatrix)-returnTarget},
                    {'type':'eq' , 'fun' : lambda x: np.sum(x) -1})
    
    bound=constraintSet
    bounds=tuple(bound for asset in range (numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def calculatedResults(meanReturns, covMatrix,riskFreeRate=0,constraintSet=(0,1)):
    #Output Max SR, Min Volatility, efficient frontier
    maxSR_Portfolio= maxSR(meanReturns, covMatrix)
    maxSR_Returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns,covMatrix)
    
    maxSR_allocation=pd.DataFrame(maxSR_Portfolio['x'],index=meanReturns.index,columns=['allocation'])
    
    
    minVar_Portfolio= minVar(meanReturns, covMatrix)
    minVar_Returns, minVar_std = portfolioPerformance(minVar_Portfolio['x'], meanReturns,covMatrix)
    
    minVar_allocation=pd.DataFrame(minVar_Portfolio['x'],index=meanReturns.index,columns=['allocation'])
    
    
    targetReturns=np.linspace(minVar_Returns,maxSR_Returns,20)
    efficientlist=[]
    for target in targetReturns:
        efficientlist.append(efficientOpt(meanReturns, covMatrix,target)['fun'])
    
    # #compare with result using 2 fund theorem
    # optPortfolios=[a*maxSR_allocation.allocation +(1-a)*minVar_allocation.allocation for a in np.linspace(0,1,20)]
    
   
    # optPortfoliosReturn=[portfolioReturn(portfolio,meanReturns,covMatrix) for portfolio in optPortfolios]
    # optPortfoliosVol=[portfolioVariance(portfolio,meanReturns,covMatrix) for portfolio in optPortfolios]
    
    
    #round all results to # to 2 dp
    efficientlist=np.array(efficientlist)
    #optPortfoliosReturn=np.array(optPortfoliosReturn)
    #optPortfoliosVol=np.array(optPortfoliosVol)
    # optPortfoliosReturn=np.round(100*optPortfoliosReturn,2)
    # optPortfoliosVol=np.round(100*optPortfoliosVol,2)
    
    maxSR_allocation.allocation=[round(i*100,1) for i in maxSR_allocation.allocation]
    minVar_allocation.allocation=[round(i*100,1) for i in minVar_allocation.allocation]
    maxSR_Returns, maxSR_std= round(100*maxSR_Returns,2), round(100*maxSR_std,2)
    minVar_Returns, minVar_std= round(100*minVar_Returns,2), round(100*minVar_std,2)
    efficientlist=np.round(100*efficientlist,2)
    targetReturns=np.round(100*targetReturns,2)
    
    return maxSR_Returns, maxSR_std, maxSR_allocation ,minVar_Returns, minVar_std, minVar_allocation , efficientlist, targetReturns 

# result= calculatedResults(meanReturns, covMatrix,riskFreeRate=0,constraintSet=(0,1))
# a,b= result[2], result[5]
# print(a+b)

def EF_graph(meanReturns, covMatrix,riskFreeRate=0,constraintSet=(0,1)):
    maxSR_Returns, maxSR_std, maxSR_allocation ,minVar_Returns, minVar_std, minVar_allocation , efficientlist, targetReturns  = calculatedResults(meanReturns, covMatrix)
    
    # Max SR
    MaxSharpeRatio= go.Scatter(
        name='Maximum Sharpe Ratio',
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_Returns],
        marker=dict(color='red',size=14,line=dict(width=3, color='black'))        
    )
    
    #Min Vol
    MinVol= go.Scatter(
        name='Minimum Volatility',
        mode='markers',
        x=[minVar_std],
        y=[minVar_Returns],
        marker=dict(color='green',size=14,line=dict(width=3, color='black'))
    )

    #efficient Frontier
    EF_Curve= go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=efficientlist,
        y=targetReturns,
        line=dict(color='black',width=4,dash='dashdot')
    )
    
    data= [MaxSharpeRatio,MinVol,EF_Curve]
    
    layout = go.Layout(
        title='Portfolio Optimisation with Efficient Frontier',
        yaxis=dict(title='Annualised Return (%)'),
        xaxis=dict(title='Annualised Volatility (%)'),
        showlegend = True,
        legend=dict(
            x=0.75 , y=0 , traceorder = 'normal',bgcolor ='#E2DFD0',
            bordercolor='black', borderwidth =2),
        width=800,
        height=600
    )
    fig=go.Figure(data=data, layout=layout)
    return fig.show()
EF_graph(meanReturns, covMatrix)

