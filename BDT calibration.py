import numpy as np
import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint
#initialise parameters
spot_market=np.array([7.3,7.62,8.1,8.45,9.2,9.64,10.12,10.45,10.75,11.22]) #market spot rates based on value of ZCB for first n integer values of t (years)
T=len(spot_market)
a=2*np.ones(T) #initialise values for a to match market values
b=0.005
q=0.5
F=100 #face value
c=0

#create rate lattice calculation based on BDT model
def rate(a,b,T): #a is array, b is scalar
  R = a[T]*np.exp(b*np.arange(T+1)) 
  return R #returns array of short term rates at time T

#pricing a bond following lattice term structure for risk free rate

def bond(F,q,a,b,t,T,c):
    c=F*c # initialise coupon payment value
    #initialise final payment array
    F=(F+c)*np.ones(T+1)
    
    for i in np.arange(T,t,-1):
        disc= 1/(1+rate(a,b,i-1)/100)
        F = disc * ( q * F[1:i+1] + (1-q) * F[0:i] )
        F+=c*np.ones(i)
        
    return F

def objective_function(a,b,T,q,F,c,spot_market):
    ZCB_model=np.zeros(T) #get solution array
    for i in range(0,len(ZCB_model)):
        ZCB_model[i]=bond(F,q,a,b,0,i+1,c) #price a ZCB expire at time i+1
        

    ZCB_model=np.array(ZCB_model/100) #as a %

    #create time array
    times=np.arange(1,T+1,1)
    #calculate model spot rates:
    spot_model=100*((1/ZCB_model)**(1/times)-1)
    target=np.sum((spot_market-spot_model)**2)
    return target


#actual[4.53,4.21,4.00,4.02,3.91,3.90,3.89,3.9,3.99,4.05]
target=objective_function(a,b,T,q,F,c,spot_market)
#print(target)

res = minimize(objective_function,a,args=(b,T,q,F,c,spot_market,))
a_cal=np.array(res.x)
print(objective_function(a_cal,b,T,q,F,c,spot_market))
print(a_cal)