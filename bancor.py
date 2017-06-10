import numpy as np
class SmartToken:
    def __init__(self,name,R,S,F,verbose=0):
        """
        Notations follow the document
        Formulas for Bancor System by Meni Rosenfeld
        """
        self.name = name
        self.R = R
        self.S = S
        self.F = F
        self.P = R/(S*F)
        self.verbose=verbose
        
    @property
    def price(self):
        return self.P
    
    @property
    def supply(self):
        return self.S
    
    @property
    def reserve(self):
        return self.R
    
    def log(self):
        if self.verbose > 0:
            print('[{}]'.format(self.name) 
                  + ' R={}'.format(np.round(self.R,2)) 
                  + ' S={}'.format(np.round(self.S,2)) 
                  + ' F={}'.format(np.round(self.F,2))
                  + ' P={}'.format(np.round(self.P,2)))
        
    def exchange_cash(self,dR):
        R0 = self.R
        S0 = self.S
        P0 = self.P
        self.R += dR
        scale = self.R/R0
        self.S = S0*np.power(scale, self.F)
        self.P = P0*np.power(scale, 1-self.F)
        dS = self.S - S0
        if self.verbose > 1:
            print('[exchange_cash] dR={dR} scale={s} P0={P0} P1={P1}'.format(
                dR=np.round(dR,2),
                s=np.round(scale,2),
                P0=np.round(P0,2),
                P1=np.round(self.P,2)
            ))
        self.log()
        return dS
    
    def exchange_shares(self,dS):
        R0 = self.R
        S0 = self.S
        P0 = self.P
        self.S += dS
        scale = self.S/S0
        self.R = R0*np.power(scale, 1/self.F)
        self.P = P0*np.power(scale, 1/self.F-1)
        dR = self.R - R0
        self.log()
        return -dR        
        
    def mkt_move(self,ret):
        scale = 1 + ret
        self.R *= scale
        self.P *= scale
        self.log()
        
class Portfolio:
    def __init__(self,cash,token,verbose=0):
        assert(cash >= 0)
        self.shares = 0
        self.cash = cash
        self.token = token
        self.verbose=verbose
        self.log()
        token.log()
        
    def log(self):
        if self.verbose > 0:
            print('[portfolio] cash={}'.format(np.round(self.cash,2))
                  + ' shares={}'.format(np.round(self.shares,2))
                  + ' wealth={}'.format(np.round(self.wealth))
                 )
        
    @property
    def wealth(self):
        return self.cash + self.shares*self.token.price
    
    def buy(self,amount):
        assert(amount >= 0)
        assert(amount <= self.cash)
        self.cash -= amount
        self.shares += self.token.exchange_cash(amount)
        self.log()
        
    def sell(self,qty):
        assert(qty >= 0)
        assert(qty <= self.shares)
        self.shares -= qty
        self.cash += self.token.exchange_shares(-qty)
        self.log()

def example1(verbose):
    print('example 1: buy then sell returns the wealth to initial level, so our intermediate wealth was "illusory"')
    BNT = SmartToken('BNT',100.0,100.0,0.2,verbose)
    pi = Portfolio(100.0,BNT,verbose)
    pi.buy(100)
    pi.sell(pi.shares)

def example2(verbose):
    print('example 2 (consistency): we check that multiple small buys are equivalent to one large one')
    BNT = SmartToken('BNT',100.0,100.0,0.2,verbose)
    pi = Portfolio(100.0,BNT,verbose)
    pi.buy(50)
    pi.buy(50)
    pi.sell(pi.shares)

def example3(verbose):
    print('example 3 (consistency): we check that if we buy and the market moves meanwhile the result is consistent')
    BNT = SmartToken('BNT',100.0,100.0,0.2,verbose)
    pi = Portfolio(100.0,BNT,verbose)
    pi.buy(50)
    BNT.mkt_move(0.05) #market moves 5% up
    pi.buy(50)
    pi.sell(pi.shares)
    np.testing.assert_almost_equal(pi.cash,102.5)
    np.testing.assert_almost_equal(BNT.price,5.25)

def main():
    verbose=1
    print('='*80)
    example1(verbose)
    print('='*80)
    example2(verbose)
    print('='*80)
    example3(verbose)

if __name__=='__main__':
    main()
