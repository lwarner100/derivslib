import datetime
import time
import json
import httplib2
import threading
import os
import pickle
import requests
import concurrent.futures as cf
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import wallstreet as ws
import yfinance as yf
import pandas.tseries.holiday

@np.vectorize
@functools.lru_cache
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

def is_trading_day(date=None):
    delta = datetime.timedelta(days=1)
    if date is not None:
        date = pd.to_datetime(date)
    else:
        date = pd.Timestamp.today()

    if hasattr(date,'__iter__'):
        result = np.isin(date,get_trading_days(start_date=np.min(date)-delta,end_date=np.max(date)+delta))
        return result if result.size > 1 else result.base[0]

    return date in get_trading_days(start_date=date-delta,end_date=date+delta)

def get_end_of_week(date=None):
    if date is not None:
        date = pd.to_datetime(date)
    else:
        date = datetime.date.today()

    days = get_trading_days(end_date=date + datetime.timedelta(7))
    return days[np.argmax([day.day_of_week for day in days])]

def get_last_trading_day(date=None):
    if date is None:
        start_date = datetime.date.today()
    else:
        start_date = pd.to_datetime(date).date()
    end_date = start_date + datetime.timedelta(days=1)
    start_date -= datetime.timedelta(days=7)
    days = get_trading_days(start_date=start_date,end_date=end_date)
    return days[-1].date()

def get_yield_curve(from_treasury=False):
    ff_url = 'https://www.federalreserve.gov/datadownload/Output.aspx?rel=PRATES&series=c27939ee810cb2e929a920a6bd77d9f6&lastobs=5&from=&to=&filetype=csv&label=include&layout=seriescolumn&type=package'
    r = requests.get(ff_url)
    ff = pd.read_csv(pd.io.common.BytesIO(r.content),header=5)
    fed_funds = ff.iloc[-1,-1] / 100

    if from_treasury:
        pardir = os.path.dirname(os.path.dirname(__file__))
        if os.path.exists(f'{pardir}/data/today_yield_curve.pkl'):
            obj = pickle.load(open(f'{pardir}/data/today_yield_curve.pkl','rb'))
            if obj.date.iloc[0] == get_last_trading_day():
                return obj
        date = datetime.date.today()
        strf = date.strftime('%Y%m')
        url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month={strf}'
        df = pd.read_html(url)[0]
        as_of_date = df.Date.iloc[-1]
        as_of_date = datetime.datetime.strptime(as_of_date,'%m/%d/%Y').date()

        cols = ['1 Mo', '2 Mo', '3 Mo', '4 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr','7 Yr', '10 Yr', '20 Yr', '30 Yr']
        vals = df[cols].iloc[-1]
        mats = np.array([float(i.split(' ')[0])/12 if 'Mo' in i else float(i.split(' ')[0]) for i in vals.index])
        yields = vals.values / 100

        result = pd.DataFrame({'mat':mats,'rate':yields,'date':as_of_date})
        ff_df = pd.DataFrame({'mat':[1/365],'rate':[fed_funds],'date':[as_of_date]})
        result = pd.concat((ff_df,result)).reset_index(drop=True)
        
        pickle.dump(result,open(f'{pardir}/data/today_yield_curve.pkl','wb'))

        return result

    url = 'https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&series=bf17364827e38702b42a58cf8eaa3f78&lastobs=5&from=&to=&filetype=csv&label=include&layout=seriescolumn'
    r = requests.get(url)
    df = pd.read_csv(pd.io.common.StringIO(r.text))
    df.columns = ['description',1/12,0.25,0.5,1.,2.,3.,5.,7.,10.,20.,30.]
    df = df[df!='ND'].dropna()

    yc_df = pd.DataFrame({'mat':df.columns[1:].values.astype(float),'rate':df.iloc[-1,1:].values.astype(float) / 100,'date':df.iloc[-1,0]})
    ff_df = pd.DataFrame({'mat':[1/365],'rate':[fed_funds],'date':[yc_df.date.iloc[0]]})
    
    result = pd.concat((ff_df,yc_df)).reset_index(drop=True)

    return result

def get_risk_free_rate(t,from_treasury=False):
    arr = False
    if hasattr(t,'__iter__'):
        t = np.array(t)
        arr = True
    curve = get_yield_curve(from_treasury=from_treasury)
    f = scipy.interpolate.interp1d(curve.mat.values,curve.rate.values,kind='cubic')
    return f(t) if arr else f(t).item()

def get_options(ticker,exp=None):
    http = httplib2.Http()
    url = f'https://query2.finance.yahoo.com/v7/finance/options/{ticker}'

    data = json.loads(http.request(url)[1])
    exps = data['optionChain']['result'][0]['expirationDates']
    compatible_dts = list(pd.to_datetime(exps,unit='s').date)

    if isinstance(exp,int):
        exps = exps[:exp]
    elif isinstance(exp,str):
        date = pd.to_datetime(exp).date()
        exps = [exps[compatible_dts.index(date)]]
    elif hasattr(exp,'year'):
        date = datetime.date(exp.year,exp.month,exp.day)
        exps = [exps[compatible_dts.index(date)]]
    elif isinstance(exp,(list, tuple, np.ndarray)):
        dates = pd.to_datetime(exp).date
        exps = [exps[compatible_dts.index(date)] for date in dates]
    else:
        pass # Use all expirations

    df = pd.DataFrame()

    def request_api(expiration):
        nonlocal ticker
        temp_http = httplib2.Http()
        date_url = f'http://query2.finance.yahoo.com/v7/finance/options/{ticker}?date={expiration}'
        data = json.loads(temp_http.request(date_url)[1])
        calls = pd.DataFrame(data['optionChain']['result'][0]['options'][0]['calls'])
        puts = pd.DataFrame(data['optionChain']['result'][0]['options'][0]['puts'])
        return pd.concat([calls,puts])

    executor = cf.ThreadPoolExecutor()
    futures = [executor.submit(request_api,exp) for exp in exps]
    df = pd.concat([f.result() for f in futures])
    
    df['contractType'] = df.contractSymbol.str[-9]

    df = (df
            .assign(
                type_sign=np.where(df.contractType == 'C', 1, -1),
                openInterest=df.openInterest.fillna(0),
                expiration=pd.to_datetime(df.expiration,unit='s').dt.date,
                lastTradeDate=pd.to_datetime(df.lastTradeDate,unit='s')
            )
        )

    return df

def get_historical_options(ticker):
    clean_ticker = ticker.replace('^','').upper()
    if ticker not in ['SPX','SPY']:
        raise ValueError('Only SPX and SPY are supported at this time')
    creds = json.load(open('./credentials.json'))
    user, token = creds.get('pythonanywhere').values()
    r = requests.get(
        f'https://www.pythonanywhere.com/api/v0/user/{user}/files/path/home/lwarner100/{clean_ticker}_options.csv',
        headers={'Authorization': f'Token {token}'}
    )

    df = pd.read_csv(pd.io.common.BytesIO(r.content))
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['date'] = pd.to_datetime(df['date'])

    return df

def put_call_ratio(ticker, exp=None):
    if isinstance(ticker, pd.DataFrame):
        options = ticker
    else:
        options = get_options(ticker,exp)
    value_counts = options.groupby(['contractType','expiration']).openInterest.sum()
    return value_counts['P'] / value_counts['C']

def option_walls(ticker,exp=3,xrange=0.1,n=5):
    options = get_options(ticker,exp)
    stock = ws.Stock(ticker)
    options = options[(options.strike > stock.price*(1-xrange))&(options.strike< stock.price*(1+xrange))]
    tbl = options.groupby(['contractType','strike']).openInterest.sum()
    return tbl.sort_values(ascending=False).groupby('contractType').head(n).sort_index(ascending=False)

def plot_option_interest(ticker,exp=None,net=False,xrange=0.2):
    options = get_options(ticker,exp)
    stock = ws.Stock(ticker)
    options = options[(options.strike > stock.price*(1-xrange))&(options.strike< stock.price*(1+xrange))]
    tbl = options.groupby(['contractType','strike']).openInterest.sum()

    if net:
        diff_tbl = tbl.C - tbl.P
        plt.bar(diff_tbl.index,diff_tbl.values,color = np.where(diff_tbl.values>0,'green','red'))
        plt.axvline(stock.price, color='black', linestyle='--')
        word = 'Net'
    else:
        plt.bar(tbl.C.index,tbl.C.values,color='green')
        plt.bar(tbl.P.index,-tbl.P.values,color='red')
        plt.axvline(stock.price, color='black', linestyle='--')
        word = ''
    plt.axhline(0, color='black')
    plt.grid()
    plt.title(f'{ticker} {word} Option Open Interest')

def get_sp500():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500

def get_bars(ticker, n_days=1, start_date=None, end_date=None, frequency=5, frequency_type='minute', after_hours=False):
    key = json.load(open('credentials.json'))['td']['key']
    base_url = f'https://api.tdameritrade.com/v1/marketdata/{ticker.upper()}/pricehistory?'
    payload = {
        'apikey':key,
        'frequencyType':frequency_type,
        'frequency':frequency
    }
    
    epoch = datetime.datetime.utcfromtimestamp(0)
    if isinstance(n_days, int) and start_date is None and end_date is None:
        payload['periodType'] = 'day'
        payload['period'] = n_days
        end_date = datetime.datetime.today().replace(hour=23,minute=59,second=59)
        payload['endDate'] = int((end_date - epoch).total_seconds() * 1000)

    if isinstance(end_date, (str, pd.Timestamp, datetime.date, datetime.datetime)):
        end_date = pd.to_datetime(end_date).replace(hour=23,minute=59,second=59)
        payload['endDate'] = int((end_date - epoch).total_seconds() * 1000)

    if end_date is None:
        end_date = datetime.datetime.today().replace(hour=23,minute=59,second=59)
        payload['endDate'] = int((end_date - epoch).total_seconds() * 1000)

    if isinstance(start_date, (str, pd.Timestamp, datetime.date, datetime.datetime)):
        start_date = pd.to_datetime(start_date).replace(hour=0,minute=0,second=0)
        payload['startDate'] = int((start_date - epoch).total_seconds() * 1000)    
    
    r = requests.get(base_url, params=payload)
    data = r.json()

    df = pd.DataFrame(data['candles'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    df.set_index('datetime', inplace=True)
    df.columns = map(str.capitalize, df.columns)
    if not after_hours:
        df = df.between_time('9:30', '15:59:59')
    
    return df

class ThreadWithReturnValue(threading.Thread):
    
    def __init__(self, group=None, target=None, name=None,args=(), kwargs={}):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,**self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return