import numpy as np
from days360 import days360
from datetime import datetime

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
time=np.concatenate(([0],time))
del_time=time[1:]-time[:-1]
del_time=1/np.sqrt(del_time)
W=np.diag(del_time)
M=np.diag(np.ones(len(time)-1))+np.diag(-np.ones(len(time)-2),-1)   

np.linalg.inv(W)
np.linalg.inv(M)
np.matmul(np.linalg.inv(M), np.linalg.inv(W))
A=np.matmul(cashflow, np.matmul(np.linalg.inv(M), np.linalg.inv(W)))
E=np.zeros(len(time)-1)
E[0]=1
RHS= price-np.matmul(cashflow,np.matmul(np.linalg.inv(M),np.transpose(E)))
delta=np.matmul(np.matmul(np.transpose(A),np.linalg.inv(np.matmul(A,np.transpose(A)))),RHS)

Zero_coupon=np.matmul(np.linalg.inv(M),np.matmul(np.linalg.inv(W),delta)+np.transpose(E))
print(time)
print(Zero_coupon)