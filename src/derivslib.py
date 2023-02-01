import os
import datetime
import warnings

import numpy as np
import scipy
import scipy.stats.qmc
import pandas as pd
import pandas.tseries.holiday

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sn

import wallstreet as ws
import yfinance as yf

import ipywidgets as widgets
from IPython.display import clear_output

from .cboe import CBOE
from . import utils

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Option:
    '''Base class for building other pricing models'''
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

    def __init__(self,*args, **kwargs):
        pass

    @staticmethod
    def date_to_t(date):
        if isinstance(date,str):
            date = pd.to_datetime(date).date()
        elif isinstance(date,datetime.datetime):
            date = date.date()

        today = pd.Timestamp.today()
        us_holidays = pd.tseries.holiday.USFederalHolidayCalendar()
        holidays = us_holidays.holidays(start=today, end=date)
        b_days = pd.bdate_range(start=today, end=date)
        trading_days = b_days.drop(holidays)
        dt = len(trading_days)
        # dt += today != trading_days[0]
        return dt/252

    @staticmethod
    def parse_symbol(symbol):
        end_sym = [char.isdigit() for idx, char in enumerate(symbol)].index(True)
        ticker = symbol[:end_sym]
        exp = datetime.datetime.strptime(symbol[end_sym:end_sym+6],'%y%m%d')
        type_ = symbol[end_sym+6:end_sym+7]
        strike = int(symbol[end_sym+7:])

        option = ws.Call(ticker,d=exp.day,m=exp.month,y=exp.year,strike=strike) if type_ == 'C' else ws.Put(ticker,d=exp.day,m=exp.month,y=exp.year,strike=strike)
        sigma = option.implied_volatility()
        div_yield = option.q
        px = round(option.underlying.price,2)
        r = 0.04

        kw = {
            's':px,
            'k':strike,
            't':exp,
            'sigma':sigma,
            'r':r,
            'type':type_,
            'style':'A',
            'qty':1,
            'q':div_yield
        }
        return kw


    def implied_volatility(self, price):
        f = lambda x: self.value(sigma = x) - price
        return scipy.optimize.newton(f, 0.3)

class BinomialOption(Option):
    '''Implementation of the Binomial Tree option pricing model
    `s`: underlying price at T=0 (now)
    `k`: strike price of the option
    `t`: the time to expiration in years or a valid date
    `sigma`: the volatility of the underlying
    `r`: the risk-free rate
    `type`: either \'call\' or \'put\' or abbrevations \'c\' or \'p\'
    `style`: either \'american\' or \'european\' or abbrevations \'a\' or \'e\'
    `n`: the number of periods to use in the binomial tree
    `qty`: the number of contracts (sign implies a long or short position)
    '''
    params = ['s','k','t','sigma','r','q','type','style','n','qty','tree']

    def __init__(self, s=100, k=100, t=1, sigma=0.3, r=0.04, type: str='C', style: str='A', n: int=500, q=0., qty: int=1):
        super().__init__()
        self.s = s
        self.k = k
        if isinstance(t,str) or isinstance(t,datetime.date) or isinstance(t,datetime.datetime):
            self.t = self.date_to_t(t)
        else:
            self.t = t
        self.sigma = sigma
        self.r = r
        self.q = q
        self.n = n
        self.qty = qty
        self.pos = 'long' if qty > 0 else 'short'

        if type not in self.valid_types.keys():
            raise ValueError('`type` must be call, C, put, or P')
        else:
            self.type = self.valid_types.get(type)
        if style not in self.valid_styles.keys():
            raise ValueError('`style` must be American, A, European, or E')
        else:
            self.style = self.valid_styles.get(style)

        self.deriv = scipy.misc.derivative

        self.get_secondary_params()

        self.default_params = {param:self.__dict__.get(param) for param in self.params}


    def __repr__(self):
        sign = '+' if self.qty > 0 else ''
        return f'{sign}{self.qty} BinomialOption(s={self.s}, k={self.k}, t={round(self.t,4)}, sigma={self.sigma}, r={self.r}, q={self.q}, type={self.type}, style={self.style})'

    def __neg__(self):
        return BinomialOption(self.s, self.k, self.t, self.sigma, self.r, type = self.type, style=self.style, n=self.n,qty=-self.qty, q=self.q)

    def __add__(self,other):
        return OptionPortfolio(self,other)

    def __sub__(self,other):
        return OptionPortfolio(self,-other)

    def __mul__(self,amount: int):
        return BinomialOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, style=self.style, n=self.n, qty=self.qty*amount, q=self.q)

    def __rmul__(self,amount: int):
        return BinomialOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, style=self.style, n=self.n, qty=self.qty*amount, q=self.q)

    @classmethod
    def from_symbol(cls, symbol, **kwargs):
        kw = cls.parse_symbol(symbol)
        kw.update(kwargs)
        return cls(**kw)

    def reset_params(self):
        self.__dict__.update(self.default_params)
        self.get_secondary_params()
        delattr(self,'tree')
        delattr(self,'value_tree')
    
    def get_secondary_params(self,trees=True):
        self.dt = self.t / self.n
        self.r_hat = np.exp((self.r-self.q)*self.dt)
        self.up = np.exp(self.sigma*np.sqrt(self.dt))
        self.dn = 1/self.up
        self.pi = (self.r_hat - self.dn)/(self.up - self.dn)
        if trees:
            self.create_tree()
            self.create_value_tree()

    def summary(self):
        data = {
                '':['price','delta','gamma','vega','theta','rho'],
                ' ':[self.price(),self.delta(),self.gamma(),self.vega(),self.theta(),self.rho()]
                }
        df = pd.DataFrame(data)
        df2 = pd.DataFrame({' ':['S','K','IV','t','r','q'],'':[self.s,self.k,self.sigma,self.t,self.r,self.q]})

        summary_df = pd.concat({'parameters':df2,'characteristics / greeks':df},axis=1)
        return summary_df

    def evaluate(self,price):
        result = np.maximum(price-self.k,0) if self.type == 'C' else np.maximum(self.k-price,0)
        return result

    def layer_evaluate(self,prices,Vu,Vd):
        vals = (1/self.r_hat) * ((self.pi*Vu) + ((1-self.pi)*Vd))
        if self.style == 'A':
            intrinsic_value = self.evaluate(prices)
            vals = np.maximum(intrinsic_value, vals)
        return vals
        
    def create_list_tree(self):
        tree = [[0]]
        for period in range(self.n):
            layer = []
            for node in tree[-1]:
                layer += [node+1,node-1]
            layer = sorted(list(set(layer)),reverse=True)
            tree.append(layer)
        tree = [[self.s*(self.up**node) for node in layer] for layer in tree]
        return tree

    def create_tree(self):
        self.tree = np.zeros((self.n+1,self.n+1))
        self.tree[0,0] = self.s
        for i in range(1,self.n+1):
            self.tree[i,0] = self.tree[i-1,0]*self.up
            self.tree[i,1:] = self.tree[i-1,:-1]*self.dn
        self.tree.sort()

    def create_trees(self):
        self.s = np.array(self.s)
        self.tree = np.zeros((len(self.s),self.n+1,self.n+1))
        self.tree[:,0,0] = self.s
        for i in range(1,self.n+1):
            self.tree[:,i,0] = self.tree[:,i-1,0]*self.up
            self.tree[:,i,1:] = self.tree[:,i-1,:-1]*self.dn
        self.tree.sort()
    
    def create_value_tree(self):
        if not hasattr(self,'tree'):
            self.create_tree()
        
        self.value_tree = np.zeros_like(self.tree)
        self.value_tree[0,:] = self.evaluate(self.tree[-1,:])
        self.value_tree[1,:-1] = self.layer_evaluate(self.tree[-2,1:] ,self.value_tree[0,1:], self.value_tree[0,:-1])
        
        for i in range(2,self.value_tree.shape[0]):
            self.value_tree[i,:-i] = self.layer_evaluate(self.tree[-i-1,i:], self.value_tree[i-1,1:-i+1], self.value_tree[i-1,:-i])
        
    def create_value_trees(self):
        if not hasattr(self,'tree'):
            self.create_trees()
        
        self.value_tree = np.zeros_like(self.tree)
        self.value_tree[:,0,:] = self.evaluate(self.tree[:,-1,:])
        self.value_tree[:,1,:-1] = self.layer_evaluate(self.tree[:,-2,1:] ,self.value_tree[:,0,1:], self.value_tree[:,0,:-1])
        
        for i in range(2,self.value_tree.shape[1]):
            self.value_tree[:,i,:-i] = self.layer_evaluate(self.tree[:,-i-1,i:] ,self.value_tree[:,i-1,1:-i+1], self.value_tree[:,i-1,:-i])

    def value(self, **kwargs):
        self.__dict__.update(kwargs)

        if kwargs:
            if hasattr(self.s,'__iter__'):
                self.get_secondary_params(trees=False)
                self.create_trees()
                self.create_value_trees()
                result = self.value_tree[:,-1,0]
                self.reset_params()
                return self.qty*result
            self.get_secondary_params()
        
        elif not hasattr(self,'value_tree'):
            self.create_value_tree()

        result = self.value_tree[-1,0]

        if kwargs:
            self.reset_params()
        
        return self.qty*result 

    def price(self):
        return self.value()

    @property
    def premium(self):
        return self.value()

    def delta(self,**kwargs):
        self.__dict__.update(kwargs)

        if kwargs:
            if hasattr(self.s,'__iter__'):
                self.create_trees()
                self.create_value_trees()
                layer = self.value_tree[:,-2]
                result = (layer[:,1]-layer[:,0]) / (self.s*(self.up-self.dn))
                self.reset_params()
                return self.qty*result
            self.get_secondary_params()
        
        elif not hasattr(self,'value_tree'):
            self.create_value_tree()

        layer = self.value_tree[-2]
        result = (layer[1]-layer[0]) / (self.s*(self.up-self.dn))

        if kwargs:
            self.reset_params()

        return self.qty*result

    def gamma(self, **kwargs):
        precision = 6e-1
        s = self.s
        if not kwargs.get('s') is None:
            s = kwargs['s']
        if hasattr(s,'__iter__'):
            s = np.array(s)
            shifts = np.sort(np.concatenate((s-precision,s+precision)))
            kwargs['s'] = shifts
            result = np.diff(self.delta(**kwargs).reshape(len(s),2),axis=1).flatten() / precision
            return result

        kwargs['s'] = np.array([s-precision,s+precision])

        result = np.diff(self.delta(**kwargs)) / (2*precision)

        return abs(self.qty)*result[0]

    def vega(self, **kwargs):
        result = self.deriv(lambda x: self.value(sigma=x, **kwargs), self.sigma, dx=1e-2)

        return abs(self.qty)*result / 100

    def theta(self, **kwargs):
        result = -self.deriv(lambda x: self.value(t=x, **kwargs), self.t, dx=1e-2)

        return abs(self.qty)*result / 365

    def rho(self, **kwargs):
        result = self.deriv(lambda x: self.value(r=x, **kwargs), self.r, dx=1e-2)
        
        return abs(self.qty)*result / 100

    def mu(self, **kwargs):
        result = self.deriv(lambda x: self.value(q=x, **kwargs), self.q, dx=1e-2)
        
        return abs(self.qty)*result / 100

    def plot(self,var='value',resolution=25, **kwargs):
        '''`var` must be either \'value\', \'delta\', \'gamma\', \'vega\', \'theta\', \'rho\', \'payoff\', or  \'pnl\''''
        greeks = {'value','delta','gamma','vega','rho','theta','pnl','payoff'}
        self.__dict__.update(kwargs)

        if kwargs:
            self.get_secondary_params()

        if var not in greeks: 
            raise ValueError('`var` must be either value, delta, gamma, vega, theta, rho, payoff, pnl')

        spot = np.linspace(self.k*0.66,self.k*1.33,resolution)

        if var == 'value' or var == 'delta' or var == 'gamma':
            vals = getattr(self,var)(s=spot)
        elif var == 'payoff':
            vals = self.qty*self.evaluate(spot)
        elif var == 'pnl':
            cost = self.value()
            vals = self.qty*self.evaluate(spot) - cost
        else:
            vals = [getattr(self,var)(s=x) for x in spot]

        plt.plot(spot,vals)
        if var == 'pnl':
            plt.title('P&L')
        else:
            plt.title(var.capitalize())
        plt.axhline(0,color='black')
        plt.axvline(self.k,linestyle='--',color='gray',alpha=0.7)

        if kwargs:
            self.reset_params()


    def plot_dist(self):
        sn.kdeplot(np.concatenate(self.create_list_tree()),fill=True)

    def show_tree(self):
        tree = self.create_list_tree()
        x = []
        for i in range(len(tree)):
            x += [i]*len(tree[i])

        ys = []
        for i in tree:
            ys += i

        plt.plot(x,ys,'o',markersize=1)
        plt.show()

class BinomialBarrierOption(BinomialOption):
    
    valid_barriers = {
        'ki':'KI',
        'ko':'KO',
        'knockin':'KI',
        'knockout':'KO'
    }

    def __init__(self, s=100, k=100, t=1, sigma=0.3, r=0.04, barrier=120, barrier_type='KI', type: str='C', style: str='A', n: int=50, qty: int = 1):
        self.barrier = barrier
        if barrier_type.lower() not in self.valid_barriers.keys():
            raise ValueError('`barrier_type` must be KI, knockin, KO, or knockout')
        else:
            self.barrier_type = self.valid_barriers.get(barrier_type.lower())
        super().__init__(s=s, k=k, sigma=sigma, t=t, r=r, n=n, type=type, style=style, qty=qty)
        

    def __repr__(self):
        sign = '+' if self.qty > 0 else ''
        return f'{sign}{self.qty} BarrierOption(s={self.s}, k={self.k}, t={round(self.t,4)}, sigma={self.sigma}, r={self.r}, barrier={self.barrier}, barrier_type={self.barrier_type}, type={self.type}, style={self.style})'

    def __neg__(self):
        return BinomialBarrierOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, style=self.style, n=self.n,qty=-self.qty, barrier=self.barrier, barrier_type=self.barrier_type)

    def __mul__(self,amount: int):
        return BinomialBarrierOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, style=self.style, n=self.n, qty=self.qty*amount, barrier=self.barrier, barrier_type=self.barrier_type)

    def __rmul__(self,amount: int):
        return BinomialBarrierOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, style=self.style, n=self.n, qty=self.qty*amount, barrier=self.barrier, barrier_type=self.barrier_type)

    def evaluate(self,price):
        conditions = {
            'C':{
                'KI':price >= self.barrier,
                'KO':price < self.barrier
            },
            'P':{
                'KI':price <= self.barrier,
                'KO':price > self.barrier
            }
        }
        return (np.maximum(price-self.k,0) if self.type == 'C' else np.maximum(self.k-price,0)) * conditions[self.type][self.barrier_type]

    def layer_evaluate(self,prices,Vu,Vd):
        vals = (1/self.r_hat) * ((self.pi*Vu) + ((1-self.pi)*Vd))
        if self.style == 'A':
            intrinsic_value = self.evaluate(prices)
            vals = np.maximum(intrinsic_value, vals)
        return vals 

class BSOption(Option):
    '''Implementation of the Black-Scholes option pricing model
    `s`: underlying price at T=0 (now)
    `k`: strike price of the option
    `t`: the time to expiration in years or a valid date
    `sigma`: the volatility of the underlying
    `r`: the risk-free rate
    `q`: the dividend yield
    `type`: either \'call\' or \'put\' or abbrevations \'c\' or \'p\'
    `qty`: the number of contracts (sign implies a long or short position)

    >>> bs_call = BSOption(s=100,k=100,t=0.25,sigma=0.3,r=0.04,type=\'call\')
    >>> bs_call.summary()
    '''
    params = ['s','k','t','sigma','r','q','type']

    def __init__(self,s=100, k=100, t=1, sigma=0.3, r=0.04, type='C', qty=1, q=0., **kwargs):
        super().__init__()
        self.s = s
        self.k = k
        if isinstance(t,str) or isinstance(t,datetime.date) or isinstance(t,datetime.datetime):
            self.t = self.date_to_t(t)
        else:
            self.t = t
        self.sigma = sigma
        self.r = r
        self.q = q
        self.qty = qty
        self.pos = 'long' if qty > 0 else 'short'
        if type not in self.valid_types.keys():
            raise ValueError('`type` must be call, C, put, or P')
        else:
            self.type = self.valid_types.get(type)

        self.price = self.value
        self.default_params = {param:self.__dict__.get(param) for param in self.params}
        self.norm_cdf = scipy.stats.norm.cdf
        self.deriv = scipy.misc.derivative

    def __neg__(self):
        return BSOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=-self.qty)

    def __add__(self,other):
        return OptionPortfolio(self,other)

    def __sub__(self,other):
        return OptionPortfolio(self,-other)

    def __mul__(self,amount: int):
        return BSOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=self.qty*amount)

    def __rmul__(self,amount: int):
        return BSOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=self.qty*amount)

    def __repr__(self):
        sign = '+' if self.qty > 0 else ''
        return f'{sign}{self.qty} BSOption(s={self.s}, k={self.k}, t={round(self.t,4)}, sigma={self.sigma}, r={self.r}, type={self.type})'

    @classmethod
    def from_symbol(cls, symbol, **kwargs):
        kw = cls.parse_symbol(symbol)
        kw.update(kwargs)
        return cls(**kw)

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
            if len(result.shape) > 1:
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

    def speed(self,**kwargs):
        '''d^3Value / ds^3 or dGamma / ds'''
        self.__dict__.update(kwargs)

        result = -(self.gamma() / self.s) * ((self.d1() / (self.sigma * np.sqrt(self.t))) + 1)

        if kwargs:
            if len(result.shape) > 1:
                result = np.diag(result)
            self.reset_params()

        return self.qty*result

    def acceleration(self,**kwargs):
        '''d^4Value / ds^4 or dSpeed / ds'''
        self.__dict__.update(kwargs)

        result = self.deriv(lambda x: self.speed(s=x), self.s, dx=1e-6, n=1)

        if kwargs:
            self.reset_params()

        return self.qty*result

    def mu(self,**kwargs):
        '''dValue / dq'''
        self.__dict__.update(kwargs)

        result = self.deriv(lambda x: self.value(q=x), self.q, dx=1e-6, n=1)

        if kwargs:
            self.reset_params()

        return self.qty*result / 100

    def jerk(self,**kwargs):
        '''d^5C / ds^5 or dAcceleration / ds'''
        self.__dict__.update(kwargs)

        result = self.deriv(lambda x: self.acceleration(s=x), self.s, dx=1e-6, n=1)

        if kwargs:
            if len(result.shape) > 1:
                result = np.diag(result)
            self.reset_params()

        return self.qty*result

    def vega(self,**kwargs):
        '''dValue / dSigma'''
        self.__dict__.update(kwargs)

        result = self.s*np.exp(-(self.d1())**2/2)/np.sqrt(2*np.pi)*np.sqrt(self.t)/100

        if kwargs:
            if len(result.shape) > 1:
                result = np.diag(result)
            self.reset_params()

        return self.qty*result

    def vanna(self,**kwargs):
        '''d^2Value / ds dSigma or dVega / ds'''
        self.__dict__.update(kwargs)

        result = (self.vega() / self.s) * (1 - (self.d1() / (self.sigma * np.sqrt(self.t))))

        if kwargs:
            if len(result.shape) > 1:
                result = np.diag(result)
            self.reset_params()

        return self.qty*result

    def theta(self,**kwargs):
        '''-dValue / dt'''
        self.__dict__.update(kwargs)

        if self.type == 'C':
            result = -self.s * scipy.stats.norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.t)) - self.r * self.k * np.exp(-self.r * self.t) * scipy.stats.norm.cdf(self.d2())
        elif self.type == 'P':
            result = -self.s * scipy.stats.norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.t)) + self.r * self.k * np.exp(-self.r * self.t) * scipy.stats.norm.cdf(-self.d2())

        result = result/365

        if kwargs:
            if len(result.shape) > 1:
                result = np.diag(result)
            self.reset_params()

        return self.qty*result

    def rho(self,**kwargs):
        '''dValue / dr'''
        self.__dict__.update(kwargs)

        if self.type == 'C':
            result = self.k * self.t * np.exp(-self.r * self.t) * scipy.stats.norm.cdf(self.d2())
        elif self.type == 'P':
            result = -self.k * self.t * np.exp(-self.r * self.t) * scipy.stats.norm.cdf(-self.d2())

        if kwargs:
            if len(result.shape) > 1:
                result = np.diag(result)
            self.reset_params()

        return self.qty*result / 100

    def summary(self):
        data = {
                '':['price','delta','gamma','vega','theta','rho'],
                ' ':[self.price(),self.delta(),self.gamma(),self.vega(),self.theta(),self.rho()]
                }
        df = pd.DataFrame(data)
        df2 = pd.DataFrame({' ':['S','K','IV','t','r','q'],'':[self.s,self.k,self.sigma,self.t,self.r,self.q]})

        summary_df = pd.concat({'parameters':df2,'characteristics / greeks':df},axis=1)
        return summary_df

    def plot(self, var='pnl', interactive=False, resolution=40, return_ax=False, **kwargs):
        '''`var` must be either \'value\', \'delta\', \'gamma\', \'vega\', \'vanna\' \'theta\', \'rho\', \'payoff\', \'pnl\', or \'summary\''''
        greeks = {'value','delta','gamma','vega','vanna','rho','theta','pnl','payoff','speed','summary'}

        x = kwargs.get('x')
        y = kwargs.get('y')
        if x and y:
            xs = np.linspace(0, 2*getattr(self,x), resolution)
            ys = getattr(self,y)(**{x:xs})
            fig, ax = plt.subplots()
            ax.plot(xs,ys)
            ax.set_xlabel(x)
            if x == 't':
                ax.invert_xaxis()
            ax.set_ylabel(y)
            return ax

        if isinstance(var,str) and var not in greeks: 
            raise ValueError('`var` must be either value, delta, gamma, speed, vega, vanna, theta, rho, payoff, pnl, or summary')

        spot = np.linspace(self.k*0.66,self.k*1.33,resolution)
        if var == 'summary':
            var = ['value','delta','gamma','vega','theta','rho']

        if hasattr(var,'__iter__') and all([i in greeks for i in var]):
            var = [i.lower() for i in var if i not in ('summary','payoff','pnl')]
            facet_map = {
                            2:(2,1),
                            3:(3,1),
                            4:(2,2),
                            5:(3,2),
                            6:(3,2)
                        }
            fig, axs = plt.subplots(facet_map.get(len(var))[1],facet_map.get(len(var))[0], figsize=(4*facet_map.get(len(var))[0],3.25*facet_map.get(len(var))[1]))
            for i, ax in enumerate(axs.flatten()):
                if i < len(var):
                    ax.plot(spot, getattr(self,var[i])(s=spot))
                    ax.set_title(var[i])
                    ax.axvline(self.k, color='black', linestyle='--', alpha=0.5)
                    ax.axhline(0, color='black')
            plt.show()
        else:
            var = var.lower()


        if (not interactive or var=='payoff') and isinstance(var,str):
            if var == 'payoff':
                vals = self.value(s=spot,t=1e-6)
            elif var == 'pnl':
                cost = self.value()
                vals = self.value(s=spot,t=1e-6) - cost
            else:
                vals = getattr(self,var)(s=spot)

            plt.plot(spot,vals)
            if var == 'pnl':
                plt.title('P&L')
            else:
                plt.title(var.capitalize())
            plt.axhline(0,color='black')
            plt.axvline(self.k,linestyle='--',color='gray',alpha=0.7)

            if return_ax:
                return plt.gca()
            else:
                plt.show()
        elif interactive and isinstance(var,str):
            def f(t=self.t,k=self.k,sigma=self.sigma,r=self.r,q=self.q):
                kwargs = {'t':t,'k':k,'t':t,'sigma':sigma,'r':r,'q':q}
                if var == 'payoff':
                    plt.plot(spot,self.value(s=spot,**kwargs),label='Value')
                    plt.plot(spot,self.value(s=spot,k=k,r=r,sigma=sigma,t=1e-6),label='Payoff at Expiration')
                elif var == 'pnl':
                    cost = self.value()
                    plt.plot(spot,self.value(s=spot,**kwargs) - cost,label='Value')
                    plt.plot(spot,self.value(s=spot,k=k,r=r,sigma=sigma,t=1e-6) - cost,label='Payoff at Expiration')
                else:
                    plt.plot(spot,getattr(self,var)(s=spot,**kwargs))

                if var == 'pnl':
                    plt.title('P&L')
                else:
                    plt.title(var.capitalize())
                plt.title(var.capitalize())
                plt.axhline(0,color='black')
                plt.axvline(k,linestyle='--',color='gray',alpha=0.7)

            interactive_plot = widgets.interactive(f, t=(0.001,2.0,0.001),sigma=(0.01,1.0,0.01), r = (0.0,0.08,0.0025), k = (self.k*0.8,self.k*1.2,0.1), q = (0,0.1,0.005))
            output = interactive_plot.children[-1]
            output.layout.height = '450px'
            return interactive_plot

class MCOption(Option):
    '''Implementation of a Monte-Carlo option pricing model
    `s`: underlying price at T=0 (now)
    `k`: strike price of the option
    `t`: the time to expiration in years or a valid date
    `sigma`: the volatility of the underlying
    `r`: the risk-free rate
    `q`: the dividend yield
    `type`: either \'call\' or \'put\' or abbrevations \'c\' or \'p\'
    `qty`: the number of contracts (sign implies a long or short position)
    `N`: the number of steps in each simulation
    `M`: the number of simulations
    `control`: the type of control variate to use. Either \'antithetic\', \'delta\', \'gamma\', or \'all\'

    >>> call = MCOption(s=100,k=100,t=0.25,sigma=0.3,r=0.04,type=\'call\',N=1,M=3e5,control=\'all\')
    >>> call.summary()

    '''
    params = ['s','k','t','sigma','r','q','type','qty','N','M','control']
    
    def __init__(self,s=100,k=100,t=1,sigma=0.3,r=0.04,type='call',q=0.,qty=1,N=1,M=20_000,control='antithetic',**kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.s = s
        self.k = k
        if isinstance(t,str) or isinstance(t,datetime.date) or isinstance(t,datetime.datetime):
            self.t = self.date_to_t(t)
        else:
            self.t = t
        self.sigma = sigma
        self.r = r
        self.q = q
        self.qty = qty
        self.N = int(N)
        self.M = int(M)
        if isinstance(control,str):
            self.control = [control]
            if control == 'all':
                self.control = ['antithetic','delta','gamma']
        elif control is None:
            self.control = []
        else:
            self.control = control
        self.pos = 'long' if qty > 0 else 'short'
        if type not in self.valid_types.keys():
            raise ValueError('`type` must be call, C, put, or P')
        else:
            self.type = self.valid_types.get(type)
        self.default_params = {param:self.__dict__.get(param) for param in self.params}
        self.deriv = scipy.misc.derivative
        self.norm_cdf = scipy.stats.norm.cdf
        self.norm_pdf = scipy.stats.norm.pdf

    def __neg__(self):
        return MCOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=-self.qty, N=self.N, M=self.M, control=self.control, q=self.q)

    def __add__(self,other):
        return OptionPortfolio(self,other)

    def __sub__(self,other):
        return OptionPortfolio(self,-other)

    def __mul__(self,amount: int):
       return MCOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=amount*self.qty, N=self.N, M=self.M, control=self.control, q=self.q)

    def __rmul__(self,amount: int):
       return MCOption(self.s, self.k, self.t, self.sigma, self.r, type=self.type, qty=amount*self.qty, N=self.N, M=self.M, control=self.control, q=self.q)

    def __repr__(self):
        sign = '+' if self.qty > 0 else ''
        return f'{sign}{self.qty} MonteCarloOption(s={self.s}, k={self.k}, t={round(self.t,4)}, sigma={self.sigma}, r={self.r}, q={self.q} type={self.type}, qty={self.qty}, N={self.N}, M={self.M})'

    @classmethod
    def from_symbol(cls, symbol, **kwargs):
        kw = cls.parse_symbol(symbol)
        kw.update(kwargs)
        return cls(**kw)

    def reset_params(self):
        self.__dict__.update(self.default_params)

    def bs_delta(self, s=None, t=None):
        if s is None:
            s = self.s
        if t is None:
            t = self.t
        d1 = (np.log(s/self.k) + (self.r + (0.5*self.sigma**2))*t)/(self.sigma*np.sqrt(t))
        if self.type == 'call':
            return self.norm_cdf(d1,0,1)
        else:
            return -self.norm_cdf(-d1,0,1)

    def bs_gamma(self,s=None,t=None):
        d1 = (np.log(s/self.k) + (self.r + 0.5*self.sigma**2)*t)/(self.sigma*np.sqrt(t))
        return np.exp(-self.r*t)*self.norm_pdf(d1,0,1)/(s*self.sigma*np.sqrt(t))

    def evaluate(self,st,k,cv_d=None):
        if 'delta' in self.control:
            self.beta1 = -1
            self.beta2 = -0.5
            rdt = np.exp((self.r)*(self.t/self.N))
            delta_St = self.bs_delta(s=st[:-1].T,t=np.linspace(self.t,0,self.N)).T
            cv_d = np.cumsum(delta_St*(st[1:] - st[:-1]*rdt), axis = 0)
            cv_g = [0]
            if self.type == 'C':
                return np.maximum(st[-1] - k, 0) + self.beta1*cv_d[-1]
            else:
                return np.maximum(k - st[-1],0) + self.beta1*cv_d[-1]
        elif 'gamma' in self.control:
            if 'delta' not in self.control:
                self.beta1 = -1
                cv_d = [0]
            self.beta2 = -0.5
            rdt = np.exp(self.r*(self.t/self.N))
            ergamma = np.exp((2*self.r+self.sigma**2)*self.dt) - 2*rdt + 1
            gamma_St = self.bs_gamma(s=st[:-1].T,t=np.linspace(self.t,0,self.N)).T
            cv_g = np.cumsum(gamma_St*((st[1:] - st[:-1])**2 - ergamma*st[:-1]**2), axis=0)
        if 'delta' in self.control or 'gamma' in self.control:
            if self.type == 'C':
                return np.maximum(st[-1] - k, 0) + self.beta1*cv_d[-1] + self.beta2*cv_g[-1]
            else:
                return np.maximum(k - st[-1],0) + self.beta1*cv_d[-1] + self.beta2*cv_g[-1]
        else:
            if self.type == 'C':
                return np.maximum(st - k,0)
            else:
                return np.maximum(k - st,0)

    def simulate(self,**kwargs):
        self.__dict__.update(kwargs)

        self.dt = self.t/self.N
        nu = (self.r-self.q) - 0.5*self.sigma**2
        qrandom = scipy.stats.qmc.Sobol(self.N,seed=0)
        Z = scipy.stats.norm.ppf(qrandom.random(self.M)).T
        # Z = np.random.normal(size=(self.N,self.M))
        log_S0 = np.log(self.s)
        log_St_delta = nu*self.dt + self.sigma*np.sqrt(self.dt)*Z
        log_St = log_S0 + np.cumsum(log_St_delta,axis=0)
        log_St = np.insert(log_St,0,log_S0,axis=0)
        St = np.exp(log_St)
        if 'antithetic' in self.control:
            log_St_delta_anti = nu*self.dt - self.sigma*np.sqrt(self.dt)*Z
            log_St_anti = log_S0 + np.cumsum(log_St_delta_anti,axis=0)
            log_St_anti = np.insert(log_St_anti,0,log_S0,axis=0)
            St_anti = np.exp(log_St_anti)
            self.paths = np.array([St, St_anti])
        else:
            self.paths = St

        if kwargs:
            self.reset_params()

    def analytics(self, **kwargs):

        begin = datetime.datetime.now()
        
        self.value(**kwargs)
        
        end = datetime.datetime.now()

        self.std_err = np.sqrt(np.sum((self.Ct[-1] - self.C0)**2)/(self.M-1)) / np.sqrt(self.M)
        self.compute_time = (end - begin).microseconds / 1000

        stats_df = pd.DataFrame({'stats':[self.qty*self.C0, self.N, self.M, self.std_err, self.compute_time]},index=['Value','N (steps)','M (simulations)','Std. Err.','Compute Time (ms)'])

        if kwargs:
            self.reset_params()

        return stats_df

    def summary(self):
        data = {
                '':['price','delta','gamma','vega','theta','rho'],
                ' ':[self.price(),self.delta(),self.gamma(),self.vega(),self.theta(),self.rho()]
                }
        df = pd.DataFrame(data)
        df2 = pd.DataFrame({' ':['S','K','IV','t','r','q'],'':[self.s,self.k,self.sigma,self.t,self.r,self.q]})

        summary_df = pd.concat({'parameters':df2,'characteristics / greeks':df},axis=1)
        return summary_df
        
    def value(self, **kwargs):
        if kwargs:
            self.__dict__.update(kwargs)
            self.M = int(self.M)
            self.N = int(self.N)

        if hasattr(self.s,'__iter__'):
            threads = []
            for spot in self.s:
                # np.random.seed(0)
                kw = kwargs.copy()
                kw['s'] = spot
                inst = self.__class__(**self.default_params)
                thread = utils.ThreadWithReturnValue(target=inst.value,kwargs=kw)
                thread.start()
                threads.append(thread)
            result = np.array([t.join() for t in threads])
            # result = np.array([self.value(s=spot, **kwargs) for spot in x])
            if kwargs:
                self.reset_params()
            return result

        self.simulate()
        if 'antithetic' in self.control:
            self.Ct = (self.evaluate(self.paths[0],self.k) + self.evaluate(self.paths[1],self.k)) / 2
        else:
            self.Ct = self.evaluate(self.paths,self.k)

        if 'delta' not in self.control:
            self.C0 = np.exp(-(self.r-self.q)*self.t)*self.Ct[-1].mean()
        else:
            self.C0 = np.exp(-(self.r-self.q)*self.t)*np.sum(self.Ct)/self.M


        if kwargs:
            self.reset_params()

        return self.C0*self.qty

    def price(self,**kwargs):
        return self.value(**kwargs)

    def delta(self, **kwargs):
        kw = kwargs.copy()
        s = kw.pop('s',self.s)

        result = self.deriv(lambda x: self.value(s=x, **kw), s, dx=1e-1)

        if kwargs:
            self.reset_params()

        return result*np.abs(self.qty)

    def gamma(self, **kwargs):
        kw = kwargs.copy()
        s = kw.pop('s',self.s)
        precision = kwargs.get('precision',6e-1)

        up, down = s-precision, s+precision
        if isinstance(up,float):
            up, down = np.array([up]), np.array([down])
        s = np.concatenate((up,down))
        s.sort()
        v = self.delta(s=s, **kw)
        result = (v[1::2]-v[::2])/(2*precision)
        if result.shape[0] == 1:
            result = result[0]

        # result = self.deriv(lambda x: self.delta(s=x, **kw), s, dx=1e-1)

        if kwargs:
            self.reset_params()

        return result*np.abs(self.qty)

    def vega(self, **kwargs):
        self.__dict__.update(kwargs)

        result = self.deriv(lambda x: self.value(sigma=x, **kwargs), self.sigma, dx=1e-2)

        if kwargs:
            self.reset_params()

        return result*self.qty / 100

    def theta(self, **kwargs):
        self.__dict__.update(kwargs)

        result = -self.deriv(lambda x: self.value(t=x, **kwargs), self.t, dx=1e-2)

        if kwargs:
            self.reset_params()

        return result*self.qty / 365
    
    def rho(self, **kwargs):
        self.__dict__.update(kwargs)

        result = self.deriv(lambda x: self.value(r=x, **kwargs), self.r, dx=1e-2)

        return result*self.qty / 100


    def plot(self, var='pnl', resolution=25, **kwargs):
        '''`var` must be either \'value\', \'delta\', \'gamma\', \'vega\', \'payoff\', \'pnl\', \'analytics\', or \'paths\'.'''
        variables = {'value','delta','pnl','payoff','paths','gamma','vega','analytics'}

        if var not in variables: 
            raise ValueError('`var` must be either value, delta, payoff, pnl')

        if var == 'analytics':
            value = self.analytics().loc['Value'].values[0]
            bs_value = BSOption(s=self.s, k=self.k, t=self.t, sigma=self.sigma, r=self.r, type=self.type, qty=self.qty).value()
            fig, ax = plt.subplots()
            c_T = np.column_stack(self.Ct)[-1] if 'antithetic' in self.control else self.Ct[-1]
            sn.kdeplot(c_T, ax=ax)
            ax.axvline(x=bs_value, color='blue', linestyle='-', label='Black-Scholes Value')
            ax.axvline(x=value, color='gray', linestyle='--', label='Estimated Value')
            ax.text(value*1.2, 0.8*ax.get_ylim()[1], f'Estimated Value: {round(value,2)}')
            ax.text(bs_value*1.2, 0.7*ax.get_ylim()[1], f'Black-Scholes Value: {round(bs_value,2)}',color='blue')
            ax.axvspan(value-self.std_err, value+self.std_err, alpha=0.2, color='gray')
            ax.set_title(f'Analytics | Std. Err.: {round(self.std_err,6)} | Compute Time: {round(self.compute_time,2)} ms')
            ax.set_xlabel('Option Value')
            ax.legend()
            return ax

        
        if var == 'paths':
            if not hasattr(self,'paths'):
                self.simulate()
            if 'antithetic' in self.control:
                plt.plot(self.paths[1],color='gray',alpha=0.025)
                plt.plot(self.paths[0],color='blue',alpha=0.025)
            else:
                plt.plot(self.paths,color='blue',alpha=0.025)
            plt.title('Simulated Stock Paths')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.show()
        else:
            spot = np.linspace(self.k*0.66,self.k*1.33,resolution)
            if var == 'payoff':
                vals = self.value(s=spot,t=1e-6)
            elif var == 'pnl':
                cost = self.value()
                vals = self.value(s=spot,t=1e-6) - cost
            elif var == 'value':
                vals = self.value(s=spot)
            else:
                vals = getattr(self,var)(s=spot)

            plt.plot(spot,vals)
            
            if var == 'pnl':
                plt.title('P&L')
            else:
                plt.title(var.capitalize())
            plt.axhline(0,color='black')
            if hasattr(self,'barrier'):
                plt.axvline(self.barrier,linestyle='--',color='gray',alpha=0.7)
            plt.axvline(self.k,linestyle='--',color='gray',alpha=0.7)
            plt.show()

class VanillaOption:
    '''
    #### Class for pricing vanilla options with the Binomial Tree, Black-Scholes, or Monte-Carlo option pricing methods
    Arguments:
    `method`: the method to use for pricing the option, either \'binomial\', \'bs\', or \'mc\'
    `s`: underlying price at T=0 (now)
    `k`: strike price of the option
    `t`: the time to expiration in years or a valid date
    `sigma`: the volatility of the underlying
    `r`: the risk-free rate
    `q`: the dividend yield
    `type`: either \'call\' or \'put\' or abbrevations \'c\' or \'p\'
    `style`: either \'american\' or \'european\' or abbrevations \'e\' or \'e\'
    `qty`: the number of contracts (sign implies a long or short position)

    Subclass specific parameters:
    Binomial Tree:
        `n`: the number of periods to use in the binomial tree
    Monte-Carlo:
        `N`: the number of steps to use in the Monte-Carlo simulation
        `M`: the number of simulations to run
        `control`: which types of control variates to apply, e.g. \'antithetic\'

    >>> call = VanillaOption(
        s = 100,
        k = 100,
        t = 0.25,
        sigma = 0.3,
        r = 0.05,
        type = \'C\',
        style = \'A\',
        n = 1000
    )
    >>> call.summary()
    '''

    valid_methods = {
        'binomial':'binomial',
        'bs':'bs',
        'black-scholes':'bs',
        'blackscholes':'bs',
        'mc':'mc',
        'monte-carlo':'mc',
        'montecarlo':'mc'
    }

    def __new__(cls, *args, **kwargs):
        method = cls.valid_methods.get(kwargs.pop('method').replace(' ','').lower())

        if method != 'binomial' and kwargs.get('style') and kwargs.get('style').lower() in ['a','american']:
            method = 'binomial'
            print('Defaulting to Binomial Tree pricing for American options')

        if method == 'mc':
            return MCOption(*args, **kwargs)
        elif method == 'bs':
            return BSOption(*args, **kwargs)
        elif method == 'binomial':
            return BinomialOption(*args, **kwargs)
        else:
            raise ValueError('Invalid method: method must be either \'binomial\', \'bs\', or \'mc\'')

class MCBarrierOption(MCOption):

    valid_barriers = {
        'ki':'KI',
        'ko':'KO',
        'knockin':'KI',
        'knockout':'KO'
    }

    params = ['s','k','t','sigma','r','type','barrier','barrier_type','qty','N','M','control']

    def __init__(self,s=100,k=100,t=1,sigma=0.3,r=0.04,type='call',qty=1, method='mc',barrier=120, barrier_type='KI', N=1, M=20_000, **kwargs):
        self.kwargs = kwargs
        self.method = method.lower()
        self.control = kwargs.get('control',[])
        super().__init__(s=s, k=k, sigma=sigma, t=t, r=r, N=N, M=M, type=type, qty=qty)
        self.M = int(M)
        self.N = int(N)
        self.s = s
        self.k = k
        if isinstance(t,str) or isinstance(t,datetime.date) or isinstance(t,datetime.datetime):
            self.t = self.date_to_t(t)
        else:
            self.t = t
        self.sigma = sigma
        self.r = r
        self.qty = qty
        self.pos = 'long' if qty > 0 else 'short'
        if type not in self.valid_types.keys():
            raise ValueError('`type` must be call, C, put, or P')
        else:
            self.type = self.valid_types.get(type)
        self.barrier = barrier
        if barrier_type.lower() not in self.valid_barriers.keys():
            raise ValueError('`barrier_type` must be KI, knockin, KO, or knockout')
        else:
            self.barrier_type = self.valid_barriers.get(barrier_type.lower())

        self.deriv = scipy.misc.derivative
        self.norm_cdf = scipy.stats.norm.cdf
        self.default_params = {param:self.__dict__.get(param) for param in self.params}

    def __repr__(self):
        sign = '+' if self.qty > 0 else ''
        return f'{sign}{self.qty} BarrierOption(s={self.s}, k={self.k}, t={round(self.t,4)}, sigma={self.sigma}, r={self.r}, barrier={self.barrier}, barrier_type={self.barrier_type}, type={self.type}, method={self.method})'

    def __neg__(self):
        return MCBarrierOption(s=self.s, k=self.k, t=self.t, sigma=self.sigma, r=self.r, type=self.type, qty=-self.qty, method=self.method, barrier=self.barrier, barrier_type=self.barrier_type,**self.kwargs)
    
    def __mul__(self,amount):
        return MCBarrierOption(s=self.s, k=self.k, t=self.t, sigma=self.sigma, r=self.r, type=self.type, qty=amount*self.qty, method=self.method, barrier=self.barrier, barrier_type=self.barrier_type,**self.kwargs)
    
    def __rmul__(self,amount):
        return MCBarrierOption(s=self.s, k=self.k, t=self.t, sigma=self.sigma, r=self.r, type=self.type, qty=amount*self.qty, method=self.method, barrier=self.barrier, barrier_type=self.barrier_type,**self.kwargs)

    def __add__(self,other):
        return OptionPortfolio(self,other)

    def __sub__(self,other):
        return OptionPortfolio(self,-other)

    def evaluate(self,st,k,cv_d=None):
        conditions = {
                    'C':{
                        'KI':st >= self.barrier,
                        'KO':st < self.barrier
                    },
                    'P':{
                        'KI':st <= self.barrier,
                        'KO':st > self.barrier
                    }
                }

        if 'delta' in self.control:
            self.beta1 = -1
            self.beta2 = -0.5
            rdt = np.exp(self.r*(self.t/self.N))
            delta_St = self.bs_delta(s=st[:-1].T,t=np.linspace(self.t,0,self.N)).T
            cv_d = np.cumsum(delta_St*(st[1:] - st[:-1]*rdt), axis = 0)
            cv_g = [0]
            if self.type == 'C':
                return np.maximum(st[-1] - k, 0) + self.beta1*cv_d[-1]
            else:
                return np.maximum(k - st[-1],0) + self.beta1*cv_d[-1]

        elif 'gamma' in self.control:
            if 'delta' not in self.control:
                self.beta1 = -1
                cv_d = [0]
            self.beta2 = -0.5
            rdt = np.exp(self.r*(self.t/self.N))
            ergamma = np.exp((2*self.r+self.sigma**2)*self.dt) - 2*rdt + 1
            gamma_St = self.bs_gamma(s=st[:-1].T,t=np.linspace(self.t,0,self.N)).T
            cv_g = np.cumsum(gamma_St*((st[1:] - st[:-1])**2 - ergamma*st[:-1]**2), axis=0)
            
        if 'delta' in self.control or 'gamma' in self.control:
            if self.barrier_type == 'KI':
                if self.type == 'C':
                    return(np.maximum((st[-1] - k), 0) + self.beta1*cv_d[-1] + self.beta2*cv_g[-1])*(st[-1] >= self.barrier)
                elif self.type == 'P':
                    return (np.maximum((k - st[-1]),0) + self.beta1*cv_d[-1] + self.beta2*cv_g[-1])*(st[-1] <= self.barrier)
            if self.barrier_type == 'KO':
                if self.type == 'C':
                    return (np.maximum((st[-1] - k), 0) + self.beta1*cv_d[-1] + self.beta2*cv_g[-1])*(st[-1] < self.barrier)
                elif self.type == 'P':
                    return (np.maximum((k - st[-1]),0) + self.beta1*cv_d[-1] + self.beta2*cv_g[-1])*(st[-1] > self.barrier)
        else:
            if self.barrier_type == 'KI':
                if self.type == 'C':
                    return np.maximum((st - k),0)*(st >= self.barrier)
                elif self.type == 'P':
                    return np.maximum((k - st),0)*(st <= self.barrier)
            elif self.barrier_type == 'KO':
                if self.type == 'C':
                    return np.maximum((st - k),0)*(st < self.barrier)
                elif self.type == 'P':
                    return np.maximum((k - st),0)*(st > self.barrier)

        '''if 'delta' in self.control or 'gamma' in self.control:
            return (np.maximum((st[-1] - k), 0) + self.beta1*cv_d[-1] + self.beta2*cv_g[-1])*combos[self.type][self.barrier_type][-1]
        else:
            return np.maximum((st - k),0)*combos[self.type][self.barrier_type]'''

class AsianOption(MCOption):


    def __init__(self, s=100,k=100,t=1,sigma=0.3,r=0.04,type='call',qty=1,N=1,M=300_000,control='antithetic',**kwargs):
        super().__init__(s=s,k=k,t=t,sigma=sigma,r=r,type=type,qty=qty,N=N,M=M,control=control,**kwargs)
        if 'delta' in self.control or 'gamma' in self.control:
            raise ValueError('Asian options can only take antithetic or no control.')

    def evaluate(self,st,k,cv_d=None):
            if self.type == 'C':
                return np.maximum(np.mean(st,axis=0) - k,0)
            else:
                return np.maximum(k - np.mean(st,axis=0),0)
        
class DeltaHedge:
    '''Represents a delta hedge of a strategy'''
    k = np.nan

    def __init__(self, **kwargs):
        self.s = kwargs.get('s')
        self.qty = kwargs.get('qty',1)
        self._gamma = kwargs.get('gamma',0)
        self._speed = kwargs.get('speed',0)
        self._acceleration = kwargs.get('acceleration',0)
        self._jerk = kwargs.get('jerk',0)

    def __repr__(self):
        args = f's={self.s}, qty={round(self.qty,3)}' if self.s and self. qty else ''
        return f'DeltaHedge({args})'

    def __neg__(self):
        return DeltaHedge(s=self.s, qty=-self.qty)

    def value(self,**kwargs):
        # result = (self.qty * kwargs.get('s') - (self.s*self.qty) if kwargs.get('s') else self.qty * self.s) - (self.s*self.qty)
        if kwargs.get('s'):
            result = (
                ( self.delta() * (kwargs.get('s') - self.s))
                + ( self.gamma() * (kwargs.get('s') - self.s)**2)
                +  ( self.speed() * (kwargs.get('s') - self.s)**3)
                +  ( self._acceleration * (kwargs.get('s') - self.s)**4)
                +  ( self._jerk * (kwargs.get('s') - self.s)**5)
                )
        else:
            result = 0
        return result

    def cost(self):
        return self.qty * self.s

    def price(self,**kwargs):
        return self.value(**kwargs)

    def delta(self,**kwargs):
        return self.qty

    def gamma(self,**kwargs):
        return self._gamma

    def speed(self,**kwargs):
        return self._speed 

    def vega(self,**kwargs):
        return 0

    def theta(self,**kwargs):
        return 0

    def rho(self,**kwargs):
        return 0

class Stock:
    
    def __init__(self, qty=1, s=100):
        self.qty = qty
        self.s = s
        self.s0 = s
        self.type = 'Stock'
        self.k = 100
        self.sigma = 0
        self.t = 0
        self.r = 0

    def __mul__(self, other):
        return Stock(qty=self.qty*other)

    def __rmul__(self, other):
        return Stock(qty=self.qty*other)

    def __neg__(self):
        return Stock(qty=-self.qty)

    def __add__(self, other):
        return OptionPortfolio(*[self,other])

    def __sub__(self, other):
        return OptionPortfolio(*[self,-other])

    def __repr__(self):
        sign = '+' if self.qty > 0 else ''
        return f'{sign}{self.qty} Stock(s={self.s})'

    def value(self, **kwargs):
        s = kwargs.get('s')
        if s is None:
            s = self.s
        return self.qty*(s-self.s0)

    def price(self,**kwargs):
        return self.value(**kwargs)

    def delta(self, **kwargs):
        return self.qty

    def gamma(self, **kwargs):
        return 0

    def vega(self, **kwargs):
        return 0

    def theta(self, **kwargs):
        return 0

    def rho(self, **kwargs):
        return 0

    def vanna(self, **kwargs):
        return 0

    def speed(self, **kwargs):
        return 0

    def jerk(self, **kwargs):
        return 0

class OptionPortfolio:
    '''A Class for holding and analyzing a portfolio of options
    `args`: a list of Option objects
    '''
    delta_hedge = None

    def __init__(self,*args,**kwargs):
        args = list(args)
        self.delta_hedge = kwargs.get('delta_hedge')
        self._mod = kwargs.get('mod')

        for idx, component in enumerate(args):
            if isinstance(component, DeltaHedge):
                self.delta_hedge = args.pop(idx)
        self.options = args

        if not np.unique([i.s for i in self.options]).size == 1:
            print('Warning: all options must have the same underlying price (using `s` of first options)')
            # raise ValueError('All options must have the same underlying price')
        self.s = max(self.options,key=lambda x: x.s)

        if self.delta_hedge:
            self.delta_hedge = DeltaHedge(s=self.s,qty=-self.delta(),gamma=-self.gamma(),speed=-self.speed(),acceleration=-self.acceleration())
            self.options.append(self.delta_hedge)

        self.ks = []
        for i in self.options:
            if not isinstance(i,Stock):
                self.ks.append(i.k)
            else:
                self.options[self.options.index(i)].s = self.s
        

    def __repr__(self):
        os = '\n'.join([repr(o) for o in self.options])
        output = f'OptionPortfolio(\n{os}\n)'
        return output

    def __neg__(self):
        return OptionPortfolio(*[-o for o in self.options])

    def __add__(self,other):
        return OptionPortfolio(*(list(self.options) + [other]))

    def __sub__(self,other):
        return OptionPortfolio(*(list(self.options) + [-other]))

    def value(self, **kwargs):
        result = sum(i.value(**kwargs) for i in self.options)

        if self._mod:
            result /= 2*self._mod

        return result

    def price(self, **kwargs):
        return self.value(**kwargs)

    def delta(self, **kwargs):
        changed_hedge = False
        if self.delta_hedge and 's' in kwargs.keys():
            changed_hedge = True
            self.options[-1] = DeltaHedge(s=kwargs['s'],qty=-sum(o.delta(s=kwargs['s']) for o in self.options[:-1]))

        result = sum(i.delta(**kwargs) for i in self.options)

        if changed_hedge:
            self.options[-1] = self.delta_hedge
        if self._mod:
                result /= 2*self._mod

        return result

    def gamma(self, **kwargs):
        result = sum(i.gamma(**kwargs) for i in self.options)
        if self._mod:
            result /= 2*self._mod
        return result

    def speed(self, **kwargs):
        result = sum(i.speed(**kwargs) for i in self.options)
        if self._mod:
            result /= 2*self._mod
        return result

    def acceleration(self, **kwargs):
        result = sum(i.acceleration(**kwargs) for i in self.options)
        if self._mod:
            result /= 2*self._mod
        return result

    def jerk(self, **kwargs):
        result = sum(i.jerk(**kwargs) for i in self.options)
        if self._mod:
            result /= 2*self._mod
        return result
    
    def vega(self, **kwargs):
        result = sum(i.vega(**kwargs) for i in self.options)
        if self._mod:
            result /= 2*self._mod
        return result

    def vanna(self, **kwargs):
        result = sum(i.vanna(**kwargs) for i in self.options)
        if self._mod:
            result /= 2*self._mod
        return result

    def theta(self, **kwargs):
        result = sum(i.theta(**kwargs) for i in self.options)
        if self._mod:
            result /= 2*self._mod
        return result

    def rho(self, **kwargs):
        result = sum(i.rho(**kwargs) for i in self.options)
        if self._mod:
            result /= 2*self._mod
        return result

    def summary(self):
        data = {
                '':['total cost','delta','gamma','vega','theta','rho'],
                ' ':[self.price(),self.delta(),self.gamma(),self.vega(),self.theta(),self.rho()]
                }
        df = pd.DataFrame(data)
        dat = {'':['value','S','K','IV','t','r']}
        format_qty = lambda x: f'+{x}' if x>0 else f'{x}'
        dat.update({f'Leg {idx+1} ({format_qty(o.qty)}{o.type})':[o.price(),o.s,o.k,o.sigma,o.t,o.r] for idx, o in enumerate(self.options) if not isinstance(o,DeltaHedge)})
        if self.delta_hedge:
            sign = '+' if self.delta_hedge.qty > 0 else ''
            dat.update({f'Delta Hedge ({sign}{round(self.delta_hedge.qty,2)} shares)':[self.delta_hedge.price(),self.delta_hedge.s,'','','','']})
        df2 = pd.DataFrame(dat)

        summary_df = pd.concat({'parameters':df2,'characteristics / greeks':df},axis=1)
        return summary_df

    def plot(self,var='pnl', interactive=False, resolution=40, xrange=0.3, **kwargs):
        '''`var` must be either \'value\', \'delta\', \'gamma\', \'vega\', \'theta\', \'rho\', \'payoff\', \'pnl\', or \'summary\''''
        greeks = {'value','delta','gamma','speed','vega','vanna','rho','theta','pnl','payoff','summary'}

        x = kwargs.get('x')
        y = kwargs.get('y')
        if x and y:
            xs = np.linspace(0, 2*getattr(self,x), resolution)
            ys = getattr(self,y)(**{x:xs})
            fig, ax = plt.subplots()
            ax.plot(xs,ys)
            ax.set_xlabel(x)
            if x == 't':
                ax.invert_xaxis()
            ax.set_ylabel(y)
            return ax

        if isinstance(var,str) and var not in greeks: 
            raise ValueError('`var` must be either value, delta, gamma, speed, vega, vanna, theta, rho, payoff, pnl, or summary')

        spot = np.linspace(min(self.ks)*(1-xrange),max(self.ks)*(1+xrange),resolution)
        spot = np.sort(np.append(spot, [np.array(self.ks)+1e-4, np.array(self.ks)-1e-4]))

        if var == 'summary':
            var = ['value','delta','gamma','vega','theta','rho']

        if hasattr(var,'__iter__') and all([i in greeks for i in var]):
            var = [i.lower() for i in var if i not in ('summary','payoff','pnl')]
            facet_map = {
                            2:(2,1),
                            3:(3,1),
                            4:(2,2),
                            5:(3,2),
                            6:(3,2)
                        }
            fig, axs = plt.subplots(facet_map.get(len(var))[1],facet_map.get(len(var))[0], figsize=(4*facet_map.get(len(var))[0],3.25*facet_map.get(len(var))[1]))
            for i, ax in enumerate(axs.flatten()):
                if i < len(var):
                    ax.plot(spot, getattr(self,var[i])(s=spot))
                    ax.set_title(var[i])
                    for k in self.ks:
                        ax.axvline(k, color='black', linestyle='--', alpha=0.5)
                    ax.axhline(0, color='black')
            plt.show()
        else:
            var = var.lower()

        if (not interactive or var=='payoff') and isinstance(var,str):
            if var == 'payoff':
                vals = self.value(s=spot,t=1e-6)
                # print(vals)
            elif var == 'pnl':
                cost = self.value() if not self.delta_hedge else self.value() + self.delta_hedge.qty*self.delta_hedge.s
                plt.plot(spot,self.value(s=spot) - cost)
                vals = self.value(s=spot,t=1e-6) - cost
            else:
                vals = getattr(self,var)(s=spot)

            plt.plot(spot,vals)
            if var == 'pnl':
                plt.title('P&L')
            else:
                plt.title(var.capitalize())
            plt.axhline(0,color='black')
            for k in self.ks:
                plt.axvline(k,linestyle='--',color='gray',alpha=0.7)

            plt.show()
            
        elif interactive and isinstance(var,str):
            def f(t=2):
                fig, ax = plt.subplots(figsize=(8,4.9))
                if var == 'payoff':
                    ax.plot(spot,self.value(s=spot,t=t),label='Value')
                    ax.plot(spot,self.value(s=spot,t=1e-6),label='Payoff at Expiration')
                    ax.legend()
                elif var == 'pnl':
                    cost = self.value() if not self.delta_hedge else self.value() + self.delta_hedge.qty*self.delta_hedge.s
                    ax.plot(spot,self.value(s=spot,t=t) - cost,label='Value')
                    ax.plot(spot,self.value(s=spot,t=1e-6) - cost,label='Payoff at Expiration')
                    ax.legend()
                else:
                    ax.plot(spot,getattr(self,var)(s=spot,t=t))

                if var == 'pnl':
                    ax.set_title('P&L')
                else:
                    ax.set_title(var.capitalize())
                ax.axhline(0,color='black')
                for k in self.ks:
                    ax.axvline(k,linestyle='--',color='gray',alpha=0.7)
                return ax

            interactive_plot = widgets.interactive(f, t=(0.001,2.0,0.01))
            output = interactive_plot.children[-1]
            output.layout.height = '450px'
            return interactive_plot

class DigitalOption(OptionPortfolio):
    '''Digital Option
    `s`: underlying price at T=0 (now)
    `k`: strike price of the option
    `t`: the time to expiration in years or a valid date
    `sigma`: the volatility of the underlying
    `r`: the risk-free rate
    `type`: either \'call\' or \'put\' or abbrevations \'c\' or \'p\'
    `qty`: the number of contracts (sign implies a long or short position)
    `precision`: the shift in `k` when calculating the limit of a spread
    '''

    def __init__(self, s=100,k=100,t=1,sigma=0.3,r=0.04,type='C',qty=1, precision=1e-6):
        type_coeff = 1 if type=='C' else -1
        self.components = [
            BSOption(s=s,k=k+(type_coeff*precision),t=t,sigma=sigma,r=r,type=type,qty=-qty),
            BSOption(s=s,k=k-(type_coeff*precision),t=t,sigma=sigma,r=r,type=type,qty=qty),
            ]
        super().__init__(*self.components,mod=precision)

        self.s = s
        self.k = k
        if isinstance(t,str) or isinstance(t,datetime.date) or isinstance(t,datetime.datetime):
            self.t = Option.date_to_t(t)
        else:
            self.t = t
        self.sigma = sigma
        self.r = r
        self.type = type
        self.qty = qty
        self.precision = precision

    def __repr__(self):
        sign = '+' if self.qty > 0 else ''
        return f'{sign}{self.qty} DigitalOption(s={self.s}, k={self.k}, t={round(self.t,4)}, sigma={self.sigma}, r={self.r}, type={self.type})'

    def __neg__(self):
        return DigitalOption(s=self.s,k=self.k,t=self.t,sigma=self.sigma,r=self.r,type=self.type,qty=-self.qty)

    def __add__(self,other):
        return OptionPortfolio(self,other)

    def __sub__(self,other):
        return OptionPortfolio(self,-other)

    def __mul__(self,amount):
        return DigitalOption(s=self.s,k=self.k,t=self.t,sigma=self.sigma,r=self.r,type=self.type,qty=self.qty*amount)

    def __rmul__(self,amount):
        return DigitalOption(s=self.s,k=self.k,t=self.t,sigma=self.sigma,r=self.r,type=self.type,qty=self.qty*amount)

class BSBarrierOption(OptionPortfolio):

    def __init__(self,s=100,k=100,t=1,sigma=0.3,r=0.04,type='call', qty=1,barrier=120, barrier_type='KI'):
        spread = abs(barrier - k)
        coeff = 1 if barrier_type == 'KI' else -1
        self.components = [
            BSOption(s, barrier, t, sigma, r, type, qty=qty),
            DigitalOption(s, barrier, t, sigma, r, type)*(spread*coeff*qty)
            ]
        if barrier_type == 'KO':
            self.components[0] = BSOption(s, k, t, sigma, r, type, qty=qty)
            self.components.append(BSOption(s, barrier, t, sigma, r, type, qty=-qty))
        super().__init__(*self.components)
        self.ks.append(k)

class BarrierOption:
    '''
    #### Class for pricing barrier options with the Binomial Tree, Black-Scholes, or Monte-Carlo option pricing methods
    Arguments:
    `method`: the method to use for pricing the option, either \'binomial\', \'bs\', or \'mc\'
    `s`: underlying price at T=0 (now)
    `k`: strike price of the option
    `t`: the time to expiration in years or a valid date
    `sigma`: the volatility of the underlying
    `r`: the risk-free rate
    `barrier`: the barrier price
    `barrier_type`: the type of barrier, \'KI\' for knock-in, \'KO\' for knock-out
    `type`: either \'call\' or \'put\' or abbrevations \'c\' or \'p\'
    `style`: either \'american\' or \'european\' or abbrevations \'e\' or \'e\'
    `qty`: the number of contracts (sign implies a long or short position)

    Subclass specific parameters:
    Binomial Tree:
        `n`: the number of periods to use in the binomial tree
    Monte-Carlo:
        `N`: the number of steps to use in the Monte-Carlo simulation
        `M`: the number of simulations to run
        `control`: which types of control variates to apply, e.g. \'antithetic\'

    >>> barrier = BarrierOption(
        s = 100,
        k = 100,
        t = 0.25,
        sigma = 0.3,
        r = 0.05,
        barrier = 80
        barrier_type = \'KI\',
        type = \'P\',
        style = \'A\'
    )
    >>> barrier.summary()
    '''
    valid_methods = {
        'binomial':'binomial',
        'bs':'bs',
        'black-scholes':'bs',
        'blackscholes':'bs',
        'mc':'mc',
        'monte-carlo':'mc',
        'montecarlo':'mc'
    }

    def __new__(cls, *args, **kwargs):
        method = cls.valid_methods.get(kwargs.pop('method').replace(' ','').lower())

        if kwargs.get('style') and kwargs.get('style').lower() in ['a','american']:
            method = 'binomial'
            print('Defaulting to binomial method for American style options')
        if method == 'mc':
            return MCBarrierOption(*args, **kwargs)
        elif method == 'bs':
            return BSBarrierOption(*args, **kwargs)
        elif method == 'binomial':
            return BinomialBarrierOption(*args, **kwargs)
        else:
            raise ValueError('Invalid method: method must be either \'binomial\', \'bs\', or \'mc\'')

class VarianceSwap(OptionPortfolio):

    def __init__(self,realized_vol,k_vol,t,r,s=100,notional=1e3,n=100):
        self.n = n
        self.s = s
        self.r = r
        self.t = t
        self.k_vol = k_vol
        self.sigma = realized_vol
        self.realized_vol = realized_vol
        self.s_cutoff = np.exp(self.r*self.t)*self.s
        self.notional = notional
        super().__init__(*[
                BSOption(
                s=self.s,
                k=k,
                t=self.t,
                sigma=self.k_vol,
                r=self.r,
                type=self.put_or_call(k),
                qty=k**-2
                )
            for k in np.linspace(1,2*s,self.n)
            ], mod=1/notional)

    def __repr__(self):
        return f'VarianceSwap(vol={self.realized_vol} ,k={self.k_vol}, t={round(self.t,4)}, s={self.s}, r={self.r}, notional={self.notional})'

    def put_or_call(self,x):
        return 'C' if x > self.s_cutoff else 'P'

    def plot(self,var='pnl', interactive=False, resolution=40, **kwargs):
        '''`var` must be either \'value\', \'delta\', \'gamma\', \'vega\', \'theta\', \'rho\', \'payoff\', \'pnl\''''
        greeks = {'value','delta','gamma','vega','rho','theta','pnl','payoff'}

        x = kwargs.get('x')
        y = kwargs.get('y')
        if x and y:
            xs = np.linspace(1e-6, 2*getattr(self,x), resolution)
            ys = getattr(self,y)(**{x:xs})
            fig, ax = plt.subplots()
            ax.plot(xs,ys)
            ax.set_xlabel(x)
            if x == 't':
                ax.invert_xaxis()
            ax.set_ylabel(y)
            ax.grid()
            return ax

        if isinstance(var,str) and var not in greeks: 
            raise ValueError('`var` must be either value, delta, gamma, vega, theta, rho, payoff, pnl')

        sigma = np.linspace(0,self.k_vol*2,resolution)

        var = var.lower()

        if (not interactive or var=='payoff') and isinstance(var,str):
            if var == 'payoff':
                vals = self.value(sigma=sigma,t=1e-6)
            elif var == 'pnl':
                cost = self.value()
                plt.plot(sigma,self.value(sigma=sigma) - cost)
                vals = self.value(sigma=sigma,t=1e-6) - cost
            else:
                vals = getattr(self,var)(sigma=sigma)

            plt.plot(sigma,vals)
            if var == 'pnl':
                plt.title('P&L')
            else:
                plt.title(var.capitalize())
            plt.axhline(0,color='black')
            plt.axvline(self.k_vol,linestyle='--',color='gray',alpha=0.7)
            plt.grid()
            plt.show()
            
        elif interactive and isinstance(var,str):
            def f(t=2):
                fig, ax = plt.subplots(figsize=(8,4.9))
                if var == 'payoff':
                    ax.plot(sigma,self.value(sigma=sigma,t=t),label='Value')
                    ax.plot(sigma,self.value(sigma=sigma,t=1e-6),label='Payoff at Expiration')
                    ax.legend()
                elif var == 'pnl':
                    cost = self.value() if not self.delta_hedge else self.value() + self.delta_hedge.qty*self.delta_hedge.s
                    ax.plot(sigma,self.value(sigma=sigma,t=t) - cost,label='Value')
                    ax.plot(sigma,self.value(sigma=sigma,t=1e-6) - cost,label='Payoff at Expiration')
                    ax.legend()
                else:
                    ax.plot(sigma,getattr(self,var)(sigma=sigma,t=t))

                if var == 'pnl':
                    ax.set_title('P&L')
                else:
                    ax.set_title(var.capitalize())
                ax.axhline(0,color='black')
                ax.axvline(self.k_vol,linestyle='--',color='gray',alpha=0.7)
                ax.grid()
                return ax

            interactive_plot = widgets.interactive(f, t=(0.001,2.0,0.01))
            output = interactive_plot.children[-1]
            output.layout.height = '450px'
            return interactive_plot

class VolSurface:
    '''Object that retrieves the volatility surface from the market for a given underlying
    `underlying`: the underlying ticker
    `moneyness`: boolean to determine whether to use abolute strikes or % moneyness
    `source`: the source of the data, either \'CBOE\' or \'wallstreet\''''

    def __init__(self, ticker, moneyness=False, source='wallstreet'):
        self.ticker = ticker
        self.moneyness = moneyness
        self.source = source
        if self.source.lower() == 'cboe':
            self.client = CBOE()
        self.underlying = ws.Stock(self.ticker)
        self.spot = self.underlying.price

    def get_data(self):
        if self.source.lower() == 'cboe':
            calls = self.client.get_options(self.ticker,'C')
            puts = self.client.get_options(self.ticker,'P')
            data = pd.concat([calls,puts])
        else:
            data = utils.get_options(self.ticker,20)
        self.data = data.rename(columns={'expiration':'expiry','impliedVolatility':'mid_iv','contractType':'option_type'})

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

        fig = go.Figure(data=[go.Mesh3d(x=self.surface.strike, y=self.surface.expiry, z=self.surface.mid_iv, intensity=self.surface.mid_iv)])

        fig.show()

    @property
    def surface_table(self):
        if not hasattr(self,'surface'):
            self.surface = self.get_vol_surface(moneyness=self.moneyness)
        return self.surface.pivot_table('mid_iv','strike','expiry').dropna()

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
        chain = utils.get_options(self.ticker,date)
        rel = chain[(chain.strike > underlying_price*0.66) & (chain.strike < underlying_price*1.33)].copy()
        rel = rel[rel.openInterest > np.quantile(rel.openInterest,quantile)]
        spot = np.linspace(underlying_price*0.66,underlying_price*1.33,50)
        spot = np.sort(np.append(spot,underlying_price))
        for index, row in rel.iterrows():
            option = BSOption(
                s = underlying_price,
                k = row.strike,
                r = 0.04,
                t = row.expiration,
                sigma = row.impliedVolatility,
                type = row.contractType
            )
            gams = option.gamma(s=spot)*row.openInterest*row.type_sign*100*underlying_price
            aggs.update({row.contractSymbol:gams})

        agg_gammas = np.nansum(list(aggs.values()), axis=0)
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

class CBOEGEX:
    '''Object that retrieves the GEX data from the market'''
    
    def __init__(self, CLIENT_ID=None, CLIENT_SECRET=None):
        try:
            self.client = CBOE()
        except ValueError:
            self.client = CBOE(CLIENT_ID,CLIENT_SECRET)
        self.today = datetime.datetime.today()
        self.spy = ws.Stock('SPY')

    def get_gex(self,date=None):
        none_date = date is None
        if date:
            if isinstance(date,str):
                if 'e' in date:
                    date = self.client.convert_exp_shorthand(date)
                else:
                    date = pd.to_datetime(date)
            month = date.month
            year = date.year
            day = date.day or None
        else:
            month = self.today.month
            year = self.today.year
            day = None

        calls = self.client.get_options('SPY','C')
        puts = self.client.get_options('SPY','P')
        data = pd.concat([calls,puts])
        if not none_date:
            query = f'exp_month == {month} and exp_year == {year}' if not day else f'exp_month == {month} and exp_year == {year} and exp_day == {day}'
            data = data.query(query)

        return data.sort_values('strike')

    def plot(self, date=None, quantile=0.7):
        sequitur = 'for' if date else 'as of'
        if not date:
            str_date = self.today.strftime('%m-%d-%Y')
        elif 'e' in date:
            date = self.client.convert_exp_shorthand(date)
            str_date = date.strftime('%m-%d-%Y')
        else:
            date = pd.to_datetime(date)
            str_date = date.strftime('%m-%d-%Y')

        gex = self.get_gex(date)
        high_interest = gex[gex.agg_gamma > gex.agg_gamma.quantile(quantile)]

        aggs = {}
        underlying_price = self.spy.price
        spot = np.linspace(underlying_price*0.66,underlying_price*1.33,50)
        for i in high_interest.iterrows():
            i = i[1]
            option = BSOption(
                s = underlying_price,
                k = i.strike,
                r = 0.04,
                t = i.expiry,
                sigma = i.mid_iv,
                type = i.option_type
            )
            gams = np.array([option.gamma(s=x)*i.open_interest*i.dealer_pos*100*underlying_price for x in spot])
            aggs.update({i.option:gams})

        agg_gammas = np.nansum(list(aggs.values()), axis=0)
        nearest_gamma = np.abs(spot - underlying_price).argmin()
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
        # add text saying the spot price in black text with white outline to the right of the point
        ax.text(underlying_price*1.02, agg_gammas[nearest_gamma], f'${underlying_price:,.2f}', ha='left', va='center', color='white', bbox=dict(facecolor='black', alpha=0.5))
        ax.legend()
        ax.grid()
        plt.show()

if __name__=='__main__':
    pass