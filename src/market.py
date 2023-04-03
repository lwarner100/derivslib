import datetime
import os
import requests
import concurrent.futures as cf
import pickle
import json
import httplib2

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import wallstreet as ws
import yfinance as yf
import pandas.tseries.holiday

from . import utils

def get_yield_curve(from_treasury=False):
    ff_url = 'https://www.federalreserve.gov/datadownload/Output.aspx?rel=PRATES&series=c27939ee810cb2e929a920a6bd77d9f6&lastobs=5&from=&to=&filetype=csv&label=include&layout=seriescolumn&type=package'
    r = requests.get(ff_url)
    ff = pd.read_csv(pd.io.common.BytesIO(r.content),header=5)
    fed_funds = ff.iloc[-1,-1] / 100

    if from_treasury:
        pardir = os.path.dirname(os.path.dirname(__file__))
        if os.path.exists(f'{pardir}/data/today_yield_curve.pkl'):
            obj = pickle.load(open(f'{pardir}/data/today_yield_curve.pkl','rb'))
            if obj.date.iloc[0] == utils.get_last_trading_day():
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
    ff_df = pd.DataFrame({'mat':[0],'rate':[fed_funds],'date':[yc_df.date.iloc[0]]})
    
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
        stock = ws.Stock(ticker)
        date = pd.to_datetime(date).date()
        days = (datetime.date.today() - date).days
        df = stock.historical(days+1)
        underlying_price = df[df.Date.dt.date == date].Close.iloc[0]
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
        gammas[option_type] = np.array([np.sum(option.gamma(s=i)) for i in spot])*100*underlying_price

    agg_gammas = gammas['C'] - gammas['P']
    nearest_gamma = np.abs(spot - underlying_price).argmin()

    return agg_gammas[nearest_gamma]

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

class VolSurface:
    '''Object that retrieves the volatility surface from the market for a given underlying
    `underlying`: the underlying ticker
    `moneyness`: boolean to determine whether to use abolute strikes or % moneyness
    `source`: the source of the data, either \'CBOE\' or \'wallstreet\''''

    def __init__(self, ticker, moneyness=False, source='wallstreet'):
        self.ticker = ticker
        self.moneyness = moneyness
        self.source = source
        self.underlying = ws.Stock(self.ticker)
        self.spot = self.underlying.price

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
        self.ws_objs = {
            self.ticker:ws.Stock(self.ticker),
            'SPY':ws.Stock('SPY'),
            '^SPX':ws.Stock('^SPX')
        }

    def bs_gamma(self, s, k, t, sigma, r):
        d1 = (np.log(s/(k*((1+r)**-t))) + ((0.5*sigma**2))*t)/(sigma*(t**0.5))
        return np.exp(-(d1)**2/2)/np.sqrt(2*np.pi)/(s*sigma*np.sqrt(t))

    def dealer_gamma(self, date=None, quantile=0.7, gamma_shifts=False):
        aggs = {}
        underlying_price = self.ws_objs[self.ticker].price
        chain = get_options(self.ticker,date)
        rel = chain[(chain.strike > underlying_price*0.66) & (chain.strike < underlying_price*1.33)].copy()
        rel = rel[rel.openInterest > np.quantile(rel.openInterest,quantile)]
        spot = np.linspace(underlying_price*0.66,underlying_price*1.33,50)
        spot = np.sort(np.append(spot,underlying_price))

        gammas = {}
        for option_type in ['C','P']:
            df = rel[rel.contractType == option_type]
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

        underlying_price = self.ws_objs[self.ticker].price
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
