import numpy as np

bond_data=[(0.5,5),(1,5.1),(2,5.32),(4,5.49),(6,5.7)]
times=[bond[0] for bond in bond_data] #extract maturity times
spot_market=[bond[1] for bond in bond_data] #extract face values

times=np.array(times)
spot_market=np.array(spot_market) #convert to numpy arrays for later use

dt=min([times[i+1]-times[i] for i in range(len(times)-1)]) #get the smallest time step for our lattice
T=max(times)/dt #get end time
print(T)

a=2*np.ones(int(T)) #initialise values for a to match market values
b=0.005
q=0.5
F=100 #face value
c=0

#create rate lattice calculation based on BDT model
def rate(a,b,T,dt): #a is array, b is scalar
  T=int(T/dt)
  R = a[T]*np.exp(b*np.arange(T+1)) 
  return R #returns array of short term rates at time T
print(rate(a,b,2,dt))

#pricing a bond following lattice term structure for risk free rate

def bond(F,q,a,b,t,T,dt,c):
    c=F*c # initialise coupon payment value
    #initialise final payment array
    T=int(T/dt)
    F=(F+c)*np.ones(T+1)
    
    for i in np.arange(T,t,-1):
        disc= 1/(1+rate(a,b,i-1,dt)/100)
        F = disc * ( q * F[1:i+1] + (1-q) * F[0:i] )
        F+=c*np.ones(i)
        
    return F
print(bond(F,q,a,b,0,0.5,dt,c))