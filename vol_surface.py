import yfinance as yf
import warnings
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm

# for interactive figures
#%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

warnings.filterwarnings("ignore")

# Get the options data for a specific stock
asset = yf.Ticker("AAPL")
expirations=asset.options
#get current price of stock and round to nearest 5 to prep for our Strike mesh
price = asset.info['currentPrice']
price=int(round(price/5.0)*5)

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
print(len(expirations_years))
print(len(Z_p))
for i in error_expirations:
    print(i)
    Z_p=np.delete(Z_p,i,0)
    expirations_years=np.delete(expirations_years,i)
print(len(expirations_years))
print(len(Z_p))   
    
# define a grid for the surface
X , Y = np.meshgrid(all_strikes, expirations_years)  

# plot the surface
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z_p, cmap=cm.coolwarm)
ax.set_ylabel('Maturity (years)') 
ax.set_xlabel('Strike') 
ax.set_zlabel('C(K, T)')
ax.set_title('Apple Call prices')
plt.savefig('fig3.png')
plt.show()
