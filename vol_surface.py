import os
import time
import pickle
import numpy as np
import pandas as pd
import concurrent.futures as cf
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, date
from thetadata import ThetaClient, OptionReqType, OptionRight, DateRange, DataType

your_username = 'featherstonenicholas@gmail.com'
your_password = 'G@ndalf41223'
def get_expirations(root_ticker) -> pd.DataFrame:
    """Request expirations from a particular options root"""
    # Create a ThetaClient
    client = ThetaClient(username=your_username, passwd=your_password, timeout=15)
    # Connect to the Terminal
    with client.connect():
        # Make the request
        data = client.get_expirations(
            root=root_ticker,
        )
    return data
root_ticker = 'SPY'
expirations = get_expirations(root_ticker)


time_now=datetime.today()
exp_dates = expirations[expirations > time_now + timedelta(days=7)]
print(exp_dates)

def get_strikes(root_ticker, expiration_date) -> pd.DataFrame:
    """Request strikes from a particular option contract"""
    # Create a ThetaClient
    client = ThetaClient(username=your_username, passwd=your_password, timeout=15)
    # Connect to the Terminal
    with client.connect():
        # Make the request
        data = client.get_strikes(
            root=root_ticker,
            exp=expiration_date
        )
    return data
root_ticker = 'SPY'
exp_date = date(2022,9,16)
strikes = get_strikes(root_ticker, exp_date)
print(strikes)

all_strikes = {}
for exp_date in exp_dates:
    all_strikes[exp_date] = pd.to_numeric(get_strikes(root_ticker, exp_date))