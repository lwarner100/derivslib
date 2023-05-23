import datetime
import requests
import concurrent.futures as cf
import json
import http
import functools

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from . import utils

@functools.lru_cache(maxsize=None)
def get_yield_curve(date=None):
    if date is None:
        end_date = datetime.date.today()
    else:
        end_date = pd.to_datetime(date).date()
    start_date = end_date - datetime.timedelta(days=4)
    start_date_str = start_date.strftime('%m/%d/%Y')
    end_date_str = end_date.strftime('%m/%d/%Y')

    fed_url = 'https://www.federalreserve.gov/datadownload/Output.aspx'
    yc_ext = f'?rel=H15&series=350823c81f512b2596ef134f67cc01dc&lastobs=&from={start_date_str}&to={end_date_str}&filetype=csv&label=omit&layout=seriescolumn'
    yc_endpoint = fed_url + yc_ext
    
    r = requests.get(yc_endpoint)
    df = pd.read_csv(pd.io.common.BytesIO(r.content), header=1)
    df = df.replace('ND', np.nan).ffill()

    df.columns = ['date',1/12,0.25,0.5,1.,2.,3.,5.,7.,10.,20.,30.,1/365]
    df = df[['date',1/365,1/12,0.25,0.5,1.,2.,3.,5.,7.,10.,20.,30.]]
    df['date'] = pd.to_datetime(df.date).dt.date
    df = pd.DataFrame({'mat':df.columns[1:].values.astype(float),'rate':df.iloc[-1,1:].values.astype(float) / 100})
    
    return df

def get_risk_free_rate(t,date=None):
    arr = False
    if hasattr(t,'__iter__'):
        t = np.array(t)
        arr = True
    curve = get_yield_curve(date)
    f = scipy.interpolate.interp1d(curve.mat.values,curve.rate.values,kind='cubic')
    return f(t) if arr else f(t).item()

def get_quote(ticker):
    url = '/v6/finance/quote'
    conn = http.client.HTTPSConnection('query2.finance.yahoo.com')
    conn.request('GET', f'{url}?symbols={ticker}')
    response = conn.getresponse()
    dat = json.load(response)

    return dat['quoteResponse']['result'][0]

@functools.lru_cache(maxsize=None)
def get_price(ticker, date=None):
    if date is None:
        return get_quote('SPY')['regularMarketPrice']
    date = pd.to_datetime(date).date()
    epoch = datetime.datetime.utcfromtimestamp(0)

    unix1 = int((utils.get_last_trading_day(date) - epoch.date()).total_seconds())
    unix2 = int((utils.get_last_trading_day(date + datetime.timedelta(4)) - epoch.date()).total_seconds())

    url = f'/v7/finance/chart/{ticker}'
    conn = http.client.HTTPSConnection('query2.finance.yahoo.com')
    conn.request('GET', f'{url}?interval=1d&period1={unix1}&period2={unix2}')
    response = conn.getresponse()
    dat = json.load(response)

    price = dat['chart']['result'][0]['indicators']['adjclose'][0]['adjclose'][0]

    return price

def get_price_history(ticker, start_date=None, end_date=None, interval='5m',extended_hours=False):
    limit_dict = {
        '5m':30,
        '1m':7
    }
    ext_bool_str = str(extended_hours).lower()

    if end_date is None:
        end_date = utils.get_last_trading_day()

    if start_date is None:
        start_date = limit_dict.get(interval, 30)

    if isinstance(start_date, int):
        if interval != '1d' and start_date > limit_dict[interval]:
            start_date = min(start_date, limit_dict[interval])
            print('Invalid number of days back passed. Defaulting to minimum safe number of days: ', start_date)
        begin_date = utils.get_last_trading_day() - datetime.timedelta(days=start_date*1.8)
        if interval == '1m':
            start_date = datetime.date.today() - datetime.timedelta(days=limit_dict[interval]-1)
        else:
            start_date = utils.get_trading_days(begin_date)[-start_date]

    epoch = datetime.datetime.utcfromtimestamp(0)
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date() + datetime.timedelta(1)

    dates = utils.get_trading_days(start_date, datetime.date.today())
    if len(dates) > limit_dict.get(interval, np.inf):
        print('Too many days passed. Defaulting to minimum safe number of days: ', limit_dict[interval])
        start_date = dates[-limit_dict[interval]]
        
    unix1 = int((start_date - epoch.date()).total_seconds())
    unix2 = int((end_date - epoch.date()).total_seconds())

    url = f'/v7/finance/chart/{ticker}'
    conn = http.client.HTTPSConnection('query2.finance.yahoo.com')
    conn.request('GET', f'{url}?interval={interval}&period1={unix1}&period2={unix2}&includePrePost={ext_bool_str}')
    response = conn.getresponse()
    r = json.load(response)


    data = {'date':r['chart']['result'][0]['timestamp']}
    data.update(r['chart']['result'][0]['indicators']['quote'][0])
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df.date, unit='s') - datetime.timedelta(hours=4)
    if interval == '1d':
        df['date'] = df['date'].dt.date
        df = df[df['date'].isin(utils.get_trading_days(start_date, end_date).date)]
    else:
        df = df[df['date'].dt.date.isin(utils.get_trading_days(start_date, end_date).date)]
    df['date'] = df.date.astype('datetime64[ns]')
    df.columns = df.columns.str.capitalize()
   
    return df

def get_options(ticker,exp=None):
    url = f'/v7/finance/options/{ticker}'
    conn = http.client.HTTPSConnection('query2.finance.yahoo.com')
    if exp is None:
        conn.request('GET', f'{url}?getAllData=true')
        response = conn.getresponse()
        data = json.load(response)['optionChain']['result'][0]['options']
        
        calls = pd.concat(pd.DataFrame(i['calls']) for i in data)
        puts = pd.concat(pd.DataFrame(i['puts']) for i in data)
        df = pd.concat([calls, puts])
    
    else:
        conn.request('GET', f'{url}')
        response = conn.getresponse()
        data = json.load(response)
        
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
            exps = compatible_dts.copy() # Use all expirations

        df = pd.DataFrame()

        def request_api(expiration):
            url = f'/v7/finance/options/{ticker}'
            conn = http.client.HTTPSConnection('query2.finance.yahoo.com')
            conn.request('GET', f'{url}?date={expiration}')
            response = conn.getresponse()
            data = json.load(response)

            calls = pd.DataFrame(data['optionChain']['result'][0]['options'][0]['calls'])
            puts = pd.DataFrame(data['optionChain']['result'][0]['options'][0]['puts'])
            return pd.concat([calls,puts])

        executor = cf.ThreadPoolExecutor()
        futures = [executor.submit(request_api,exp) for exp in exps]
        df = pd.concat([f.result() for f in futures])
        
    df['contractType'] = df.contractSymbol.str[-9]
    df['expiration'] = pd.to_datetime(df.expiration,unit='s').dt.date

    df = (df
            .assign(
                type_sign=np.where(df.contractType == 'C', 1, -1),
                openInterest=df.openInterest.fillna(0),
                lastTradeDate=pd.to_datetime(df.lastTradeDate,unit='s')
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

def dealer_gamma(data, ticker, date, quantile=0.7, r=None):
    '''Get the dealer gamma for a given ticker and date given a dataset of options
    \nInteresting use:
    >>> dealer_gammas = df.groupby('date').apply(lambda x: dealer_gamma(x, ticker, x.date.iloc[0]))
    '''
    if 'Close' not in data.columns:
        date = pd.to_datetime(date).date()
        underlying_price = get_price(ticker, date)
    else:
        underlying_price = data[data.date == date].Close.iloc[0]
    chain = data
    rel = chain[(chain.strike > underlying_price*0.66) & (chain.strike < underlying_price*1.33)].copy()
    rel = rel[rel.openInterest > np.quantile(rel.openInterest,quantile)]
    spot = np.linspace(underlying_price*0.66,underlying_price*1.33,50)
    spot = np.sort(np.append(spot,underlying_price))

    gammas = {}
    for option_type in ['C','P']:
        df = rel[rel.contractType == option_type]
        k = df.strike.values
        t = utils.date_to_t(df.expiration.values,t0=date)
        sigma = df.impliedVolatility.values
        r = 0.04 if r is None else r # get_risk_free_rate(t)
        option = LiteBSOption(
            s = underlying_price,
            k = k,
            r = r,
            t = t,
            sigma = sigma,
            type = option_type,
            qty=df.openInterest.values
        )
        gammas[option_type] = np.sum(option.gamma(s=underlying_price))*100*underlying_price

    agg_gammas = gammas['C'] - gammas['P']

    return agg_gammas

def gamma_walls(df, ticker, date=None):
    if date is None:
        date = utils.get_last_trading_day()

    if 'date' in df.columns:
        df = df[df.date.dt.date == date].copy()

    underlying_price = get_price(ticker, date)
    rel = df[(df.strike > underlying_price*0.66) & (df.strike < underlying_price*1.33)].copy()
    spot = np.linspace(underlying_price*0.66,underlying_price*1.33,250)
    spot = np.sort(np.append(spot,underlying_price))
    gammas = {}

    for option_type in ['C','P']:
        df = rel[rel.contractType == option_type].copy()
        k = df.strike.values
        t = utils.date_to_t(df.expiration.values,t0=date)
        sigma = df.impliedVolatility.values
        r = get_risk_free_rate(t)
        option = LiteBSOption(
            s = underlying_price,
            k = k,
            r = r,
            t = t,
            sigma = sigma,
            type = option_type,
            qty=df.openInterest.values
        )
        gammas[option_type] = np.array([np.sum(option.gamma(s=i)) for i in spot])*100*spot

    agg_gammas = gammas['C'] - gammas['P']
    max_gamma = spot[agg_gammas.argmax()]
    min_gamma = spot[agg_gammas.argmin()]
    zero_gamma = spot[np.abs(agg_gammas).argmin()]

    return {'max':max_gamma,'min':min_gamma,'zero':zero_gamma}

def option_walls(ticker,exp=3,xrange=0.1,n=5):
    options = get_options(ticker,exp)
    price = get_price(ticker)
    options = options[(options.strike > price*(1-xrange))&(options.strike< price*(1+xrange))]
    tbl = options.groupby(['contractType','strike']).openInterest.sum()
    return tbl.sort_values(ascending=False).groupby('contractType').head(n).sort_index(ascending=False)

def plot_option_interest(ticker,exp=None,net=False,xrange=0.2):
    options = get_options(ticker,exp)
    price = get_price(ticker)
    options = options[(options.strike > price*(1-xrange))&(options.strike< price*(1+xrange))]
    tbl = options.groupby(['contractType','strike']).openInterest.sum()

    if net:
        diff_tbl = tbl.C - tbl.P
        plt.bar(diff_tbl.index,diff_tbl.values,color = np.where(diff_tbl.values>0,'green','red'))
        plt.axvline(price, color='black', linestyle='--')
        word = 'Net'
    else:
        plt.bar(tbl.C.index,tbl.C.values,color='green')
        plt.bar(tbl.P.index,-tbl.P.values,color='red')
        plt.axvline(price, color='black', linestyle='--')
        word = ''
    plt.axhline(0, color='black')
    plt.grid()
    plt.title(f'{ticker} {word} Option Open Interest')

class VolSurface:
    '''Object that retrieves the volatility surface from the market for a given underlying
    `underlying`: the underlying ticker
    `moneyness`: boolean to determine whether to use abolute strikes or % moneyness
    `source`: the source of the data, either \'CBOE\' or \'wallstreet\''''

    def __init__(self, ticker, moneyness=False):
        self.ticker = ticker
        self.moneyness = moneyness
        self.spot = get_price(ticker)

    def get_data(self):
        data = get_options(self.ticker,20)
        self.data = data.rename(columns={'expiration':'expiry','impliedVolatility':'mid_iv','contractType':'option_type'})
        self.data = self.data[-(self.data.lastTradeDate - datetime.datetime.today()).dt.days < 5]
        return self.data

    def get_vol_surface(self,moneyness=False):
        if not hasattr(self,'data'):
            self.data = self.get_data()

        vol_data = self.data[['strike','mid_iv','expiry','option_type']]
        vol_data = vol_data[((vol_data.strike >= self.spot)&(vol_data.option_type=='C'))|((vol_data.strike < self.spot)&(vol_data.option_type=='P'))]
        if moneyness:
            vol_data['strike'] = (vol_data.strike / self.spot)*100
        
        self.surface = vol_data.sort_values(['expiry','strike'])
        
        return self.surface

    def skew_plot(self,*args):
        if not hasattr(self,'surface'):
            self.surface = self.get_vol_surface(moneyness=self.moneyness)

        idx = 0
        if args:
            idx = [int(i) for i in args if str(i).isnumeric()]

        tbl = self.surface.pivot_table('mid_iv','strike','expiry').dropna()
        tbl.iloc[:,idx].plot()
        ttl = tbl.columns[idx][0].strftime('Expiration: %m-%d-%Y') if idx!=0 else tbl.columns[idx].strftime('Expiration: %m-%d-%Y')
        plt.title(ttl)
        if self.moneyness:
            plt.xlabel('strike (% moneyness)')
        plt.plot()

    def surface_plot(self):
        if not hasattr(self,'surface'):
            self.surface = self.get_vol_surface(moneyness=self.moneyness)

        # fig = go.Figure(data=[go.Mesh3d(x=self.surface.strike, y=self.surface.expiry, z=self.surface.mid_iv, intensity=self.surface.mid_iv)])
        fig = go.Figure(data=[go.Surface(x=self.surface_table.columns, y=self.surface_table.index, z=self.surface_table.values)])

        fig.show()

    @property
    def surface_table(self):
        if not hasattr(self,'surface'):
            self.surface = self.get_vol_surface(moneyness=self.moneyness)
        return self.surface.pivot_table('mid_iv','strike','expiry').dropna(axis=0)

class GEX:

    def __init__(self,ticker: str = 'SPY'):
        self.ticker = ticker
        self.today = datetime.datetime.today()
        self.spot = get_price(ticker)

    def bs_gamma(self, s, k, t, sigma, r):
        d1 = (np.log(s/(k*((1+r)**-t))) + ((0.5*sigma**2))*t)/(sigma*(t**0.5))
        return np.exp(-(d1)**2/2)/np.sqrt(2*np.pi)/(s*sigma*np.sqrt(t))

    def dealer_gamma(self, date=None, quantile=0.7, gamma_shifts=False):
        aggs = {}
        underlying_price = self.spot
        chain = get_options(self.ticker,date)
        rel = chain[(chain.strike > underlying_price*0.66) & (chain.strike < underlying_price*1.33)].copy()
        # rel = rel[rel.openInterest > np.quantile(rel.openInterest,quantile)]
        spot = np.linspace(underlying_price*0.66,underlying_price*1.33,50)
        spot = np.sort(np.append(spot,underlying_price))
        gammas = {}
        for option_type in ['C','P']:
            df = rel[rel.contractType == option_type].copy()
            k = df.strike.values
            t = utils.date_to_t(df.expiration.values)
            sigma = df.impliedVolatility.values
            r = get_risk_free_rate(t)
            option = LiteBSOption(
                s = underlying_price,
                k = k,
                r = r,
                t = t,
                sigma = sigma,
                type = option_type,
                qty=df.openInterest.values
            )
            gammas[option_type] = np.array([np.sum(option.gamma(s=i)) for i in spot])*100*underlying_price

        agg_gammas = gammas['C'] - gammas['P']
        nearest_gamma = np.abs(spot - underlying_price).argmin()

        if gamma_shifts:
            return agg_gammas, nearest_gamma

        return agg_gammas[nearest_gamma]

    def plot(self, date=None, quantile=0.7):
        sequitur = 'as of' if date is None or isinstance(date,int) else 'for'
        if date is None:
            date = utils.get_next_opex()
            str_date = date.strftime('%m-%d-%Y')
        elif isinstance(date,int):
            str_date = self.today.strftime('%m-%d-%Y')
        elif 'e' in date:
            date = utils.convert_exp_shorthand(date)
            str_date = date.strftime('%m-%d-%Y')
        else:
            date = pd.to_datetime(date)
            str_date = date.strftime('%m-%d-%Y')

        underlying_price = self.spot
        spot = np.linspace(underlying_price*0.66,underlying_price*1.33,50)
        spot = np.sort(np.append(spot,underlying_price))
        
        agg_gammas, nearest_gamma = self.dealer_gamma(date,quantile,gamma_shifts=True)
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(spot, agg_gammas, label='Dealer Gamma')
        ax.set_xlim(spot[0],spot[-1])
        ax.vlines(underlying_price,0,agg_gammas[nearest_gamma],linestyle='--',color='gray')
        ax.hlines(agg_gammas[nearest_gamma],spot[0],underlying_price,linestyle='--',color='gray')
        ax.plot(underlying_price, agg_gammas[nearest_gamma], 'o', color='black', label='Spot')
        ax.set_title(f'Dealer Gamma Exposure {sequitur} {str_date}')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Gamma Exposure')
        ax.axhline(0,color='black')
        ax.text(underlying_price*1.02, agg_gammas[nearest_gamma], f'${underlying_price:,.2f}', ha='left', va='center', color='white', bbox=dict(facecolor='black', alpha=0.5))
        ax.legend()
        ax.grid()
        plt.show()

class LiteOption:
    valid_types = {
        'c':'C',
        'C':'C',
        'Call':'C',
        'call':'C',
        'p':'P',
        'P':'P',
        'Put':'P',
        'put':'P'
    }

    valid_styles = {
        'a':'A',
        'A':'A',
        'American':'A',
        'american':'A',
        'e':'E',
        'E':'E',
        'European':'E',
        'european':'E',
    }

    default_params = {}

    def __init__(self,*args, **kwargs):
        self.date_to_t = utils.date_to_t

    def update(self, inplace=True,**kwargs):
        if not inplace:
            params = self.default_params.copy()
            params.update(kwargs)
            return self.__class__(**params)
        self.default_params.update(kwargs)
        self.reset_params()

    def reset_params(self):
        self.__dict__.update(self.default_params)

    def implied_volatility(self, price, guess=0.3):
        f = lambda x: self.value(sigma = x) - price
        if isinstance(price, np.ndarray):
            shape = price.shape
            f = lambda x: (self.value(sigma = x.reshape(shape), synced=False) - price).flatten()
            guess_arr = np.full_like(price, guess).flatten()
            return scipy.optimize.newton(f, guess_arr, maxiter=100_000_000, tol=1e-10).reshape(shape)
        return scipy.optimize.newton(f, guess, maxiter=10000)


class LiteBSOption(LiteOption):
    params = ['s','k','t','sigma','r','q','type']

    def __init__(self,s=100, k=100, t=1, sigma=0.3, r=None, type='C', qty=1, q=0., **kwargs):
        super().__init__()
        self.s = s
        self.k = k
        if isinstance(t,(str,datetime.date,datetime.datetime)):
            self.t = self.date_to_t(t)
        else:
            self.t = t
        self.sigma = sigma
        self.r = r if r is not None else get_risk_free_rate(self.t)
        self.q = q
        self.qty = qty
        if type not in self.valid_types.keys():
            raise ValueError('`type` must be call, C, put, or P')
        else:
            self.type = self.valid_types.get(type)

        self.price = self.value
        self.default_params = {param:self.__dict__.get(param) for param in self.params}
        self.norm_cdf = scipy.special.ndtr #scipy.stats.norm.cdf
        self.deriv = scipy.misc.derivative

    def __neg__(self):
        return LiteBSOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=-self.qty)

    def __mul__(self,amount: int):
        return LiteBSOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=self.qty*amount)

    def __rmul__(self,amount: int):
        return LiteBSOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=self.qty*amount)

    def __repr__(self):
        sign = '+' if self.qty > 0 else ''
        return f'{sign}{self.qty} BSOption(s={self.s}, k={self.k}, t={round(self.t,4)}, sigma={self.sigma}, r={self.r}, type={self.type})'

    def reset_params(self):
        self.__dict__.update(self.default_params)

    def d1(self):
        return (np.log((np.exp(-self.q*self.t)*self.s)/(self.k*np.exp(-self.r*self.t))) + ((0.5*self.sigma**2))*self.t)/(self.sigma*(self.t**0.5))

    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.t)
    
    def value(self,**kwargs):
        self.__dict__.update(kwargs)
        
        if self.type == 'C':
            result = (np.exp(-self.q*self.t)*self.s)*self.norm_cdf(self.d1()) - self.k*np.exp(-self.r*self.t)*self.norm_cdf(self.d2())
        elif self.type == 'P':
            result = self.k*np.exp(-self.r*self.t)*self.norm_cdf(-self.d2()) - (np.exp(-self.q*self.t)*self.s)*self.norm_cdf(-self.d1())

        if kwargs:
            if len(result.shape) > 1 and kwargs.get('synced',True):
                result = np.diag(result)
            self.reset_params()

        return self.qty*result
    
    @property
    def premium(self):
        return self.value()

    def delta(self,**kwargs):
        '''dValue / ds'''
        self.__dict__.update(kwargs)
        
        result = self.norm_cdf(self.d1())
        if self.type == 'P':
            result -= 1

        if kwargs:
            if len(result.shape) > 1:
                result = np.diag(result)
            self.reset_params()

        return self.qty*result
        
    
    def gamma(self,**kwargs):
        '''d^2Value / ds^2 or dDelta / ds'''
        self.__dict__.update(kwargs)

        result = np.exp(-(self.d1())**2/2)/np.sqrt(2*np.pi)/(self.s*self.sigma*np.sqrt(self.t))

        if kwargs:
            if len(result.shape) > 1:
                result = np.diag(result)
            self.reset_params()

        return self.qty*result
