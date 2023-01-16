import os
import requests
import datetime
import base64
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wallstreet as ws

if os.path.exists('credentials.json'):
    with open('credentials.json', 'r') as f:
        credentials = json.load(f).get('accounts')[0]
else:
    credentials = {}

class CBOE:

    def __init__(self, CLIENT_ID=credentials.get('client_id'), CLIENT_SECRET=credentials.get('client_secret')):
        if not CLIENT_ID and not CLIENT_SECRET:
            raise ValueError('Could not read credentials from local credentials.txt. Please input valid credentials as arguments.')

        self.client_id = CLIENT_ID
        self.client_secret = CLIENT_SECRET
        self.connect()
        self.today = datetime.datetime.today()


    def connect(self):
        identity_url = "https://id.livevol.com/connect/token"
        authorization_token  = base64.b64encode((self.client_id + ':' + self.client_secret).encode())
        headers = {"Authorization": "Basic " + authorization_token.decode('ascii')}
        payload = {"grant_type": "client_credentials"}

        token_data = requests.post(identity_url, data=payload, headers=headers)

        if token_data.status_code == 200:
            self.access_token = token_data.json()['access_token']
            if not len(self.access_token) > 0:
                print('Authentication failed')
        else:
            print("Authentication failed:",token_data.content)        

    def convert_exp_shorthand(self,date: str):
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
            return self.get_opex_date(month,year)
        else:
            raise ValueError('Invalid date format - must be in the format {month}e{year}')


    @staticmethod
    def get_opex_date(month,year):
        d = datetime.date(year,month,1)
        d += datetime.timedelta( (4-d.weekday()) % 7 )
        d += datetime.timedelta(14)
        return d

    @staticmethod
    def dealer_pos(option_type):
        if option_type == 'C':
            return 1
        elif option_type == 'P':
            return -1

    @staticmethod
    def date_to_t(date, start_date=None):
        if isinstance(date,str):
            date = pd.to_datetime(date).date()
        elif isinstance(date,datetime.datetime):
            date = date.date()

        today = start_date or pd.Timestamp.today()
        us_holidays = pd.tseries.holiday.USFederalHolidayCalendar()
        holidays = us_holidays.holidays(start=today, end=date)
        dt = len(pd.bdate_range(start=today, end=date)) - len(holidays)
        
        return dt/252

    def bs_gamma(self, s, k, t, sigma, r):
        d1 = (np.log(s/(k*((1+r)**-t))) + ((0.5*sigma**2))*t)/(sigma*(t**0.5))
        return np.exp(-(d1)**2/2)/np.sqrt(2*np.pi)/(s*sigma*np.sqrt(t))

    def get_quote(self, ticker, option_type='C', date=None):
        today = date or self.today.strftime('%Y-%m-%d')
        if isinstance(today,(datetime.date, datetime.datetime)):
            today = today.strftime('%Y-%m-%d')
        
        max_exp = (self.today + datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        
        
        self.stock = ws.Stock(ticker)
        spot = self.stock.price
        min_k = int(spot * 0.7)
        max_k = int(spot * 1.3)

        url = f'https://api.livevol.com/v1/live/allaccess/market/option-and-underlying-quotes?root={ticker}&option_type={option_type}&date={today}&min_expiry={today}&max_expiry={max_exp}&min_strike={min_k}&max_strike={max_k}&symbol={ticker}'
        headers = {'Authorization': 'Bearer ' + self.access_token}
        data = requests.get(url, headers=headers)
        
        return data
        
    def get_options(self, ticker, option_type='C', date=None):
        r = self.get_quote(ticker,option_type,date)
        df = pd.DataFrame(r.json().get('options'))
        df['expiry'] = pd.to_datetime(df.expiry)
        df['dealer_pos'] = df.option_type.apply(self.dealer_pos)
        df = df.assign(
            exp_month = df.expiry.dt.month,
            exp_year = df.expiry.dt.year,
            exp_day = df.expiry.dt.day,
            agg_gamma = df.gamma * df.open_interest,
            dealer_gamma = df.gamma * df.open_interest * df.dealer_pos * 100 * self.stock.price
        )

        return df
            
    def get_dealer_gamma(self,date,spy_price):
        month = date.month
        year = date.year
        day = date.day
        underlying_price = spy_price

        opex = self.get_opex_date(month,year)
        if  opex < date:
            fwd_date = date + datetime.timedelta(days=30)
            opex = self.get_opex_date(fwd_date.month,fwd_date.year)

        query = f'exp_month == {opex.month} and exp_year == {opex.year}' if not day else f'exp_month == {opex.month} and exp_year == {opex.year} and exp_day == {opex.day}'
        calls = self.get_options('SPY','C',date=date)
        puts = self.get_options('SPY','P',date=date)
        data = pd.concat([calls,puts])
        data['as_of'] = date
        data = data.query(query).sort_values('strike')

        high_interest = data[data.agg_gamma > data.agg_gamma.quantile(0.7)]

        aggs = {}
        spot = np.linspace(underlying_price*0.66,underlying_price*1.33,50)
        for index, i in high_interest.iterrows():
            gams = np.array(self.bs_gamma(s=underlying_price, k=i.strike, t=self.date_to_t(i.expiry.date(),start_date=i.as_of), sigma=i.mid_iv, r=0.02)*i.open_interest*i.dealer_pos*100*underlying_price)
            aggs.update({i.option:gams})

        agg_gammas = np.nansum(list(aggs.values()), axis=0)
        nearest_gamma = np.abs(spot - underlying_price).argmin()

        return nearest_gamma

    