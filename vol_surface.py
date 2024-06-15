import yfinance as yf
import warnings
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import vega
# for interactive figures
#%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

warnings.filterwarnings("ignore")
def getcalldata(stock):
    
    # Get the options data for a specific stock
    asset = yf.Ticker(stock)
    expirations=asset.options
    #get current price of stock and round to nearest 5 to prep for our Strike mesh
    S0 = asset.info['currentPrice']
    price=int(round(S0/5.0)*5)
    #get long name
    name=asset.info['longName']
    #create a list of strikes around current price
    all_strikes = np.arange(price-50, price +55, 5)

    #create a list of expirations in years
    expirations_years=[]
    Z_p = np.empty([len(expirations), len(all_strikes)])

    error_expirations=[]
    for i in range(len(expirations)):
        expiration=expirations[i]
        opt=asset.option_chain(expiration)
        calls= opt.calls
    
        calls['mid'] = calls[['bid','ask']].mean(axis=1)
        price=calls['mid']
        s=calls['strike']
        sMin=min(s)
        
        #get the maturity in years
        T=(pd.to_datetime(expiration)-dt.datetime.today()).days/365
        expirations_years.append(T)
        f = interpolate.interp1d(s, price, bounds_error=False, fill_value=0)
        Z_p[i, :] = f(all_strikes) 
    
        #check for expirations which will create holes in our mesh
        if sMin > min(all_strikes):
            error_expirations.append(i) 
    print('Number of error maturities: '+str(len(error_expirations))) 
    for i in error_expirations:
        Z_p=np.delete(Z_p,i,0)
        expirations_years=np.delete(expirations_years,i)
        expirations=np.delete(expirations,i)
    return name,S0,all_strikes, expirations, expirations_years, Z_p
def implied_vol(S0, K, T, r, market_price, flag='c',tol=0.00001):
    #calculate implied volatility of european option
    max_iter = 200 # max number of iterations
    vol_old=0.3 #first guess
    
    for k in range(max_iter):
        bs_price=bs(flag,S0,K,T,r,vol_old)
        c_prime=vega(flag,S0,K,T,r,vol_old)*100
        
        vol_new=vol_old-(bs_price-market_price)/c_prime
        new_bs_price=bs(flag,S0,K,T,r,vol_new)
        if(abs(vol_old-vol_new)<tol or abs(new_bs_price-market_price)<tol):
            break
        
        vol_old=vol_new
    implied_vol=vol_new
    return implied_vol

name, S0, all_strikes, expirations, expirations_years, market_prices =getcalldata('AAPL') 
# define a grid for the surface
X , Y = np.meshgrid(all_strikes, expirations_years)  

# plot the surface
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, market_prices, cmap=cm.coolwarm)
ax.set_ylabel('Maturity (years)') 
ax.set_xlabel('Strike') 
ax.set_zlabel('C(K, T)')
ax.set_title(name +' Call prices')
# plt.savefig('fig3.png')
# plt.show()
def vol_surface(all_strikes,expirations_years,market_prices,r):
    impVolMat = np.empty([len(expirations_years), len(all_strikes)])
    for i in range(len(expirations_years)):
        for j in range(len(all_strikes)):
            T=expirations_years[i]
            K=all_strikes[j]
            market_price=market_prices[i,j]
            impVolMat[i,j]=implied_vol(S0, K, T, r, market_price, flag='c',tol=0.00001)
    return impVolMat
impVolMat=vol_surface(all_strikes,expirations_years,market_prices,r=0.05)

# plot the surface
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, impVolMat, cmap=cm.coolwarm)
ax.set_ylabel('Maturity (years)') 
ax.set_xlabel('Strike') 
ax.set_zlabel('Vol')
ax.set_title(name +' Call Implied Volatility')
# plt.savefig('fig3.png')
# plt.show()

pd_market=pd.DataFrame(market_prices,expirations_years,all_strikes)
pd_market.index.name = 'Maturities (years)'
print(pd_market.head())