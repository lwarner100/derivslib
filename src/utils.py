import datetime
import threading
import functools

import numpy as np
import pandas as pd
import pandas.tseries.holiday
    
def date_to_t_minutes(date, t0=None):
    if isinstance(date,str):
        if ':' in date:
            date = pd.to_datetime(date)
        else:
            date = pd.to_datetime(date).replace(hour=16, minute=0, second=0, microsecond=0)
    elif isinstance(date,datetime.date):
        date = pd.to_datetime(date).replace(hour=16, minute=0, second=0, microsecond=0)

    if isinstance(t0,str):
        if ':' in t0:
            t0 = pd.to_datetime(t0)
        else:
            t0 = pd.to_datetime(t0).replace(hour=9, minute=30, second=0, microsecond=0)
    elif isinstance(t0,datetime.date):
        t0 = pd.to_datetime(t0).replace(hour=9, minute=30, second=0, microsecond=0)

    today = pd.Timestamp.now() if t0 is None else t0
    end_today = today.replace(hour=16, minute=0, second=0, microsecond=0)
    start_today = today.replace(hour=9, minute=30, second=0, microsecond=0)
    market_open = today > start_today and is_trading_day(today.date())
    if market_open:
        dt_to_close = today.replace(hour=16, minute=0, second=0, microsecond=0) - today
        dt_to_close = max(0, dt_to_close.total_seconds() / 60)
    
    us_holidays = pd.tseries.holiday.USFederalHolidayCalendar()
    holidays = us_holidays.holidays(start=today, end=date)
    minutes_per_day = 6.5*60

    days = len(pd.bdate_range(start=today, end=date)) - len(holidays) - market_open
    days_minutes = days*minutes_per_day
    minutes = dt_to_close if market_open else 0
    dt = (days_minutes + minutes) / minutes_per_day
    
    return dt/252

@np.vectorize
@functools.lru_cache(maxsize=None)
def date_to_t(date, t0=None, precision='day'):
    if 'm' in precision.lower():
        return date_to_t_minutes(date,t0)

    if isinstance(date,str):
        date = pd.to_datetime(date).date()
    elif isinstance(date,datetime.datetime):
        date = date.date()

    today = pd.Timestamp.today() if t0 is None else pd.to_datetime(t0).date()
    us_holidays = pd.tseries.holiday.USFederalHolidayCalendar()
    holidays = us_holidays.holidays(start=today, end=date)
    dt = len(pd.bdate_range(start=today, end=date)) - len(holidays)
    return dt/252


@np.vectorize
@functools.lru_cache(maxsize=None)
def t_to_date(t, t0=None):
    if t0 is None:
        t0 = datetime.date.today()
    elif isinstance(t0,str):
        t0 = pd.to_datetime(t0).date()
    elif isinstance(t0,datetime.datetime):
        t0 = t0.date()

    days = int(t*252)
    trading_days = get_trading_days(t0, t0 + datetime.timedelta(days*8/5))
    return trading_days[days].date()


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
        fwd_date = date + datetime.timedelta(days=28)
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
    end_date = start_date
    start_date -= datetime.timedelta(days=7)
    if is_trading_day(end_date):
        return end_date
    days = get_trading_days(start_date=start_date,end_date=end_date)
    return days[-1].date()


def get_sp500():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500

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