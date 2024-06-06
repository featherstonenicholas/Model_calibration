"""Calculate implied volatility of an option using Newton Raphson
"""
import numpy as np
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import vega

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

S0 =195.9
K = 200
T = 1.02
r = 0.0509
market_price = 21.2

print(implied_vol(S0, K, T, r, market_price, flag='c',tol=0.00001)*100)

    