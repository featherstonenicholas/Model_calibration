import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib inline
# for interactive figures
#%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy.stats import norm
from scipy import optimize
import cmath
import math

#define functions needed to price using Heston model
def generic_CF(u, params, S0, r, q, T, model):
    
    if (model == 'GBM'):
        
        sig = params[0]
        mu = np.log(S0) + (r-q-sig**2/2)*T
        a = sig*np.sqrt(T)
        phi = np.exp(1j*mu*u-(a*u)**2/2)
        
    elif(model == 'Heston'):
        
        kappa  = params[0]
        theta  = params[1]
        sigma  = params[2]
        rho    = params[3]
        v0     = params[4]
        
        tmp = (kappa-1j*rho*sigma*u)
        g = np.sqrt((sigma**2)*(u**2+1j*u)+tmp**2)
        
        pow1 = 2*kappa*theta/(sigma**2)
        
        numer1 = (kappa*theta*T*tmp)/(sigma**2) + 1j*u*T*r + 1j*u*math.log(S0)
        log_denum1 = pow1 * np.log(np.cosh(g*T/2)+(tmp/g)*np.sinh(g*T/2))
        tmp2 = ((u*u+1j*u)*v0)/(g/np.tanh(g*T/2)+tmp)
        log_phi = numer1 - log_denum1 - tmp2
        phi = np.exp(log_phi)
        
        #g = np.sqrt((kappa-1j*rho*sigma*u)**2+(u*u+1j*u)*sigma*sigma)
        #beta = kappa-rho*sigma*1j*u
        #tmp = g*T/2
        
        #temp1 = 1j*(np.log(S0)+(r-q)*T)*u + kappa*theta*T*beta/(sigma*sigma)
        #temp2 = -(u*u+1j*u)*v0/(g/np.tanh(tmp)+beta)
        #temp3 = (2*kappa*theta/(sigma*sigma))*np.log(np.cosh(tmp)+(beta/g)*np.sinh(tmp))
        
        #phi = np.exp(temp1+temp2-temp3);
        

    elif (model == 'VG'):
        
        sigma  = params[0];
        nu     = params[1];
        theta  = params[2];

        if (nu == 0):
            mu = math.log(S0) + (r-q - theta -0.5*sigma**2)*T
            phi  = math.exp(1j*u*mu) * math.exp((1j*theta*u-0.5*sigma**2*u**2)*T)
        else:
            mu  = math.log(S0) + (r-q + math.log(1-theta*nu-0.5*sigma**2*nu)/nu)*T
            phi = np.exp(1j*u*mu)*((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)**(-T/nu))

    return phi

def genericFFT(params, S0, K, r, q, T, alpha, eta, n, model):
    
    N = 2**n
    
    # step-size in log strike space
    lda = (2*np.pi/N)/eta
    
    #Choice of beta
    #beta = np.log(S0)-N*lda/2
    beta = np.log(K)
    
    # forming vector x and strikes km for m=1,...,N
    km = np.zeros((N))
    xX = np.zeros((N))
    
    # discount factor
    df = math.exp(-r*T)
    
    nuJ = np.arange(N)*eta
    psi_nuJ = generic_CF(nuJ-(alpha+1)*1j, params, S0, r, q, T, model)/((alpha + 1j*nuJ)*(alpha+1+1j*nuJ))
    
    for j in range(N):  
        km[j] = beta+j*lda
        if j == 0:
            wJ = (eta/2)
        else:
            wJ = eta
        xX[j] = cmath.exp(-1j*beta*nuJ[j])*df*psi_nuJ[j]*wJ
     
    yY = np.fft.fft(xX)
    cT_km = np.zeros((N))  
    for i in range(N):
        multiplier = math.exp(-alpha*km[i])/math.pi
        cT_km[i] = multiplier*np.real(yY[i])
    
    return km, cT_km
def eValue(params,marketPrices,maturities,strikes,r,q,S0,alpha,eta,n,model):
    lenT = len(maturities)
    lenK = len(strikes)

    modelPrices = np.zeros((lenK, lenT))
    #print(marketPrices.shape)

    count = 0
    mae = 0
    for i in range(lenT):
        for j in range(lenK):
            count  = count+1
            T = maturities[i]
            K = strikes[j]
            
            km, cT_km = genericFFT(params, S0, K, r, q, T, alpha, eta, n, model)
            modelPrices[i][j] = cT_km[0]
            
            tmp = marketPrices[i][j]-modelPrices[i][j]
            mae += tmp**2
    #print(modelPrices)    
    
    rmse = math.sqrt(mae/count)
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
    return name, S0, all_strikes, expirations, expirations_years, Z_p

name, S0, strikes, expirations, maturities_years, marketPrices =getcalldata('AAPL') 


# Parameters
alpha = 1.5
eta = 0.2
    
n = 12

# Model
model = 'Heston' 


# risk free rate
r = 0.0245
# dividend rate
q = 0.005

#parameter sets for initial parameters (Kappa, Theta, Sigma, Rho, v0)

params1 = (1.0, 0.02, 0.05, -0.4, 0.08)
params2 = (3.0, 0.06, 0.10, -0.6, 0.04)
iArray = []
rmseArray = []
rmseMin = 1e10

rmse=mfc.eValue(params1,marketPrices,maturities_years,strikes,r,q,S0,alpha,eta,n,model)