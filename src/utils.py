import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wallstreet as ws
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

def get_options(ticker, exp=None):
    if isinstance(exp,int):
        date = get_end_of_week()
        d, m, y = date.day, date.month, date.year
        call = ws.Call(ticker,d=d,m=m,y=y)
        exps = call.expirations[:exp]
        return pd.concat([get_options(ticker, datetime.datetime.strptime(day,'%d-%m-%Y')) for day in exps])
    elif exp is not None:
        exp = pd.to_datetime(exp)
        d, m, y = exp.day, exp.month, exp.year
    else:
        date = get_end_of_week()
        d, m, y = date.day, date.month, date.year

    call = ws.Call(ticker,d=d,m=m,y=y)
    put = ws.Put(ticker,d=d,m=m,y=y)


    calls = pd.DataFrame(call.data)
    calls['contractType'] = 'C'
    puts = pd.DataFrame(put.data)
    puts['contractType'] = 'P'
    options = calls.append(puts)

    options = (options
                .assign(
                    type_sign=np.where(options.contractType == 'C', 1, -1),
                    openInterest=options.openInterest.fillna(0),
                    expiration=options.expiration.apply(datetime.datetime.fromtimestamp),
                    lastTradeDate=options.lastTradeDate.apply(datetime.datetime.fromtimestamp)
                ))

    return options

def put_call_ratio(ticker, exp=None):
    if isinstance(ticker, pd.DataFrame):
        options = ticker
    else:
        options = get_options(ticker,exp)
    value_counts = options.groupby('contractType').openInterest.sum()
    return value_counts['P'] / value_counts['C']