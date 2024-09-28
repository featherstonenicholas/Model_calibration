import numpy as np
from days360 import days360
from datetime import datetime
from scipy.stats import *
from scipy.optimize import *

def timeconversion(spot_date,date):
    spot_date=datetime.strptime(spot_date,'%d/%m/%Y')
    return days360(spot_date, datetime.strptime(date,'%d/%m/%Y'), method="US")/360
def bond_cash_flow(spot_date,next_coupon,maturity,coupon_rate,annual_coupon,nominal):

    next_coupon=timeconversion(spot_date,next_coupon)
    maturity=timeconversion(spot_date,maturity)
    time_range=np.arange(next_coupon,maturity+coupon_rate,coupon_rate)
    c_flow=np.ones(len(time_range))*annual_coupon*coupon_rate*nominal
    c_flow[-1]+=nominal
    return time_range,c_flow
def swap_cash_flow(spot_date, start_date, maturity, swap_rate, fixed_payment_rate):
    
    start_date=timeconversion(spot_date,start_date)
    maturity=timeconversion(spot_date,maturity)
    time_range=np.arange(start_date+fixed_payment_rate,maturity+fixed_payment_rate,fixed_payment_rate)
    c_flow=np.ones(len(time_range))*swap_rate*fixed_payment_rate
    c_flow[-1]+=1
    price=1
    if 0<start_date:
        price = 0
        time_range=np.concatenate((time_range,[start_date]))
        c_flow=np.concatenate((c_flow,[-1]))
    return time_range,c_flow,price
    
spot_date='05/09/1996'
#input bonds in format [spot date, first coupon date, maturity, coupon rate, annual coupon,nominal,price]
bonds=[ 
['04/09/1996','15/11/1996','15/11/1996',0.5,0.1000,100,103.82],
['04/09/1996','19/01/1997','19/01/1998',0.5,0.0975,100,106.04],
['04/09/1996','26/09/1996','26/03/1999',0.5,0.1225,100,118.44],
['04/09/1996','03/03/1997','03/03/2000',0.5,0.0900,100,106.28],
['04/09/1996','06/11/1996','06/11/2001',0.5,0.0700,100,101.15],
['04/09/1996','27/02/1997','27/08/2002',0.5,0.0975,100,111.06],
['04/09/1996','07/12/1996','07/12/2005',0.5,0.0850,100,106.24],
['04/09/1996','08/03/1997','08/09/2006',0.5,0.0775,100,98.49],
['04/09/1996','13/10/1996','13/10/2008',0.5,0.0900,100,110.87], ]
bonds=[]
#input swaps in format [spot date, start date, maturity, swap rate, fixed payment rate]
swaps=[
[spot_date,'05/09/1996','05/09/1997',0.0036,0.5],
[spot_date,'05/09/1996','05/09/1998',0.0052,0.5],
[spot_date,'05/09/1996','05/09/1999',0.0093,0.5],
[spot_date,'05/09/1996','05/09/2000',0.0121,0.5],
[spot_date,'05/09/1996','05/09/2001',0.0146,0.5],
[spot_date,'05/09/1996','05/09/2002',0.0166,0.5],
[spot_date,'05/09/1996','05/09/2003',0.0184,0.5],
[spot_date,'05/09/1996','05/09/2004',0.0199,0.5],
[spot_date,'05/09/1996','05/09/2005',0.0213,0.5],
[spot_date,'05/09/1996','05/09/2006',0.0221,0.5],
[spot_date,'05/09/1996','05/09/2011',0.0263,0.5],
[spot_date,'05/09/1996','05/09/2016',0.0273,0.5],
[spot_date,'05/09/1996','05/09/2026',0.0271,0.5]
]

flows=[]
price=[]
for b in bonds:
    flows.append(bond_cash_flow(b[0],b[1],b[2],b[3],b[4],b[5]))
    price.append(b[6])
for s in swaps:
    swap=swap_cash_flow(s[0], s[1], s[2], s[3], s[4])
    flows.append((swap[0],swap[1]))
    price.append(swap[2])
#create the vector of times
time=[]
for f in flows:
    time=np.concatenate((time,f[0]))
time=np.unique(time)
#create the cash flow matrix
cashflow=[]
for f in flows:
    bond_flow=np.zeros(len(time))

    for i in range(len(f[0])):
        bond_flow[np.where(time==f[0][i])[0][0]]=f[1][i]
    cashflow.append(bond_flow)
cashflow=np.array(cashflow)


#Create the coponents for pseudo inverse method
del_time=np.concatenate(([0],time))
del_time=del_time[1:]-del_time[:-1]
del_time=1/np.sqrt(del_time)
W=np.diag(del_time)
M=np.diag(np.ones(len(time)))+np.diag(-np.ones(len(time)-1),-1)   

np.linalg.inv(W)
np.linalg.inv(M)
np.matmul(np.linalg.inv(M), np.linalg.inv(W))
A=np.matmul(cashflow, np.matmul(np.linalg.inv(M), np.linalg.inv(W)))
E=np.zeros(len(time))
E[0]=1
RHS= price-np.matmul(cashflow,np.matmul(np.linalg.inv(M),np.transpose(E)))
delta=np.matmul(np.matmul(np.transpose(A),np.linalg.inv(np.matmul(A,np.transpose(A)))),RHS)

Zero_coupon=np.matmul(np.linalg.inv(M),np.matmul(np.linalg.inv(W),delta)+np.transpose(E))
print(time)
print(Zero_coupon)
def forward_rates(time,Zero_coupon):
    time=np.concatenate(([0],time))    
    Zero_coupon=np.concatenate(([1],Zero_coupon))
    del_t=time[1:]-time[:-1]
    fwds=(1/del_t)*(Zero_coupon[:-1]/Zero_coupon[1:]-1)
    return fwds
fwds=forward_rates(time,Zero_coupon)
#last section relates to cap price calculation

def forward_swap(ZeroBondPrices,bond_time,delta,Maturity):
    swap_times=np.arange(delta,Maturity+delta,delta)
    
    time_index=[]
    for i in range(len(swap_times)):
        time_index.append(np.where(bond_time==swap_times[i])[0][0])
    kappa = (ZeroBondPrices[0] - ZeroBondPrices[int(time_index[-1])]) / (delta*sum(ZeroBondPrices[int(time_index[1]):int(time_index[-1])+1]))
   
   
    return kappa
kappa=forward_swap(Zero_coupon,time,0.5,30)
def myIntegral(b,nu,t0,t1):
    return nu**2/b**2 *(np.exp(-b*t0) - np.exp(-b*t1))**2 * (np.exp(2*b*t0)-1)/(2*b)
def CapHJM(b,nu,k,T,M,delta,Z):
    #time and discounts dont include t=0
    myAns = 0
    jump=int(1/delta)
    for i in range(1, (jump*M)):
        I = myIntegral(b[0], nu[0], T[i-1], T[i])+myIntegral(b[1], nu[1], T[i-1], T[i])
        
        d1 = (np.log(Z[i]/Z[i-1]*(1+delta*k)) + 0.5*I)/np.sqrt(I)
        d2 = (np.log(Z[i]/Z[i-1]*(1+delta*k)) - 0.5*I)/np.sqrt(I)
        cplt_i = Z[i-1]*norm.cdf(-d2,0,1)-(1+delta*k)*Z[i]*(norm.cdf(-d1,0,1))
        #print(Z[i+2])
        #print(cplt_i)
        myAns = myAns + cplt_i 
    return myAns
cap_price=CapHJM([0.3,0.5],[0.01,0.02],kappa,time,30,0.5,Zero_coupon)
print('Price of Cap using HJM model is ',CapHJM([0.3,0.5],[0.01,0.02],kappa,time,30,0.5,Zero_coupon))
#for the following, time and dsicounts include t=o and P(0,0)
def BlackCap(sig,k,fwds,T,M,delta,Z):
    myAns = 0
    
    jump=int(1/delta)
    
    for i in range(1, (jump*M)):
        d1 = (np.log(fwds[i]/k) + 0.5*(sig**2)*T[i])/(sig*np.sqrt(T[i]))
        d2 = d1 - sig*np.sqrt(T[i])
        
        cplt_i = delta*Z[i+1]*(fwds[i]*norm.cdf(d1,0,1) - k*norm.cdf(d2,0,1))
        #print('i, fwds[i], k, T[i-1], T[i], sig', i, fwds[i], k, T[i-1], T[i], sig)
        #print('d1, d2, cplt_i', d1, d2, cplt_i)
        myAns = myAns + cplt_i
    return myAns
def BachCap(sig,k,fwds,T,M,delta,Z):
    myAns = 0
    
    jump=int(1/delta)
    for i in range(1, (jump*M)):
        D= (fwds[i]-k)/(sig*np.sqrt(T[i]))
       
        cplt_i = delta*Z[i+1]*sig*np.sqrt(T[i])*(D*norm.cdf(D,0,1) +norm.pdf(D,0,1))
        
        myAns = myAns + cplt_i
    return myAns
def black_imp_vol(ForwardRates,T0,m,delta,kappa,ZeroBondPrices,CapPrice):
    BCap = lambda iv: BlackCap(iv,kappa,ForwardRates,T0,m,delta,ZeroBondPrices)-CapPrice
    imp=bisect(BCap,0.005,10)
    return imp
def bach_imp_vol(ForwardRates,T0,M,delta,k,ZeroBondPrices,CapPrice):
    BCap = lambda iv: BachCap(iv,k,ForwardRates,T0,M,delta,ZeroBondPrices)-CapPrice
    imp=bisect(BCap,0.005,10)
    return imp
#add t=o elements to time and discount vector to line up with functions from stochastic section
time=np.concatenate(([0],time))    
Zero_coupon=np.concatenate(([1],Zero_coupon))
print('Black implied volatility',black_imp_vol(fwds,time,30,0.5,kappa,Zero_coupon,cap_price))
print('Bachelier implied volatility',bach_imp_vol(fwds,time,30,0.5,kappa,Zero_coupon,cap_price))
