import datetime
import time
import json
import httplib2
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wallstreet as ws
import yfinance as yf
import pandas.tseries.holiday

def date_to_t(date):
    if isinstance(date,str):
        date = pd.to_datetime(date).date()
    elif isinstance(date,datetime.datetime):
        date = date.date()

    today = pd.Timestamp.today()
    us_holidays = pd.tseries.holiday.USFederalHolidayCalendar()
    holidays = us_holidays.holidays(start=today, end=date)
    dt = len(pd.bdate_range(start=today, end=date)) - len(holidays)
    return dt/252

def convert_exp_shorthand(date: str):
    month_map = {
        'jan':1,
        'feb':2,
        'mar':3,
        'apr':4,
        'may':5,
        'jun':6,
        'jul':7,
        'aug':8,
        'sep':9,
        'oct':10,
        'nov':11,
        'dec':12
    }
    if date[0] == 'e':
        month = month_map[date[1:4].lower()]
        year = int(date[4:])
        if len(str(year)) == 2:
            year = 2000 + year
        return get_opex_date(month,year)
    else:
        raise ValueError('Invalid date format - must be in the format e{month}{year}')


def get_opex_date(month,year):
    d = datetime.date(year,month,1)
    d += datetime.timedelta( (4-d.weekday()) % 7 )
    d += datetime.timedelta(14)
    return d

def get_next_opex(date=None):
    if date is not None:
        date = pd.to_datetime(date)
    else:
        date = datetime.date.today()
    opex = get_opex_date(date.month,date.year)
    if  opex < date:
        fwd_date = date + datetime.timedelta(days=30)
        opex = get_opex_date(fwd_date.month,fwd_date.year)

    return opex

def get_trading_days(start_date=None,end_date=None):
    if start_date == None:
        start_date = pd.Timestamp.today()
    elif end_date == None:
        end_date = pd.Timestamp.today()

    us_holidays = pd.tseries.holiday.USFederalHolidayCalendar()
    holidays = us_holidays.holidays(start=start_date, end=end_date)
    business_days = pd.bdate_range(start=start_date, end=end_date)
    return business_days.drop(holidays)

def get_end_of_week(date=None):
    if date is not None:
        date = pd.to_datetime(date)
    else:
        date = datetime.date.today()

    days = get_trading_days(end_date=date + datetime.timedelta(7))
    return days[np.argmax([day.day_of_week for day in days])]

def get_options(ticker,exp=None):
    http = httplib2.Http()
    url = f'https://query2.finance.yahoo.com/v7/finance/options/{ticker}'

    data = json.loads(http.request(url)[1])
    exps = data['optionChain']['result'][0]['expirationDates']
    compatible_dts = [datetime.date.fromtimestamp(i) + datetime.timedelta(1) for i in exps]

    if isinstance(exp,int):
        exps = exps[:exp]
    elif isinstance(exp,str):
        date = pd.to_datetime(exp).date()
        exps = [exps[compatible_dts.index(date)]]
    elif hasattr(exp,'year'):
        date = datetime.date(exp.year,exp.month,exp.day)
        exps = [exps[compatible_dts.index(date)]]
    elif isinstance(exp,(list, tuple, np.ndarray)):
        dates = [i.date() for i in pd.to_datetime(exp)]
        exps = [exps[compatible_dts.index(date)] for date in dates]
    # else:
        # date = dl.utils.get_end_of_week()
        # exps = [int(time.mktime(date.timetuple()))]

    df = pd.DataFrame()
    for expiration in exps:
        date_url = f'https://query2.finance.yahoo.com/v7/finance/options/{ticker}?date={expiration}'
        data = json.loads(http.request(date_url)[1])
        calls = pd.DataFrame(data['optionChain']['result'][0]['options'][0]['calls'])
        puts = pd.DataFrame(data['optionChain']['result'][0]['options'][0]['puts'])
        df = pd.concat([df,calls,puts])

    df['contractType'] = df.contractSymbol.str[-9]

    df = (df
            .assign(
                type_sign=np.where(df.contractType == 'C', 1, -1),
                openInterest=df.openInterest.fillna(0),
                expiration=pd.to_datetime(df.contractSymbol.str[-15:-9], format='%y%m%d')
            )
        )

    return df

def put_call_ratio(ticker, exp=None):
    if isinstance(ticker, pd.DataFrame):
        options = ticker
    else:
        options = get_options(ticker,exp)
    value_counts = options.groupby(['contractType','expiration']).openInterest.sum()
    return value_counts['P'] / value_counts['C']

def get_sp500():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500

class ThreadWithReturnValue(threading.Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return