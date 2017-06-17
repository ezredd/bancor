import numpy as np


class Reserve:
    def __init__(self, name, amount, crr, verbose=0):
        assert(amount > 0)
        assert(crr > 0.0 and crr <= 1.0)
        self.name = name
        self.amount = amount
        self.crr = crr
        self.verbose = verbose

    def __str__(self):
        if self.verbose > 0:
            return '[{}'.format(self.name) \
                  + ' R={}'.format(np.round(self.amount, 3)) \
                  + ' F={}]'.format(np.round(self.crr, 3))
        else:
            return ''

    def log(self):
        print(self)



class SmartToken2:
    def __init__(self, name, supply, reserve_tokens=dict(), verbose=0):
        self.name = name
        self.reserve_tokens = reserve_tokens
        self.supply = supply
        self.verbose=verbose

    def log(self):
        if self.verbose > 0:
            print('[{}]'.format(self.name)
                  + ' S={} :'.format(np.round(self.supply, 3)) \
                  + ':'.join([str(r) for _, r in self.reserve_tokens.items()])
            )

    def add_reserve(self, reserve):
        assert(type(reserve) is Reserve)
        assert(reserve.name not in self.reserve_tokens.keys())
        self.reserve_tokens[reserve.name] = reserve
        crr = self.crr
        assert(crr >= 0 and crr <= 1.0)

    @property
    def crr(self):
        return np.sum([r.crr for _,r in self.reserve_tokens.items()])

    @property
    def reserve(self):
        reserve_tot = np.sum([r.amount for _,r in self.reserve_tokens.items()])
        assert(reserve_tot >= 0)
        return reserve_tot

    @property
    def price(self):
        return self.reserve/(self.crr * self.supply)

    def partial_price(self, token):
        """
        Implied price of a reserve token
        :param token:
        :return:
        """
        assert(token in self.reserve_tokens.keys())
        reserve = self.reserve_tokens[token]
        assert(type(reserve) is Reserve)
        return reserve.amount / (reserve.crr * self.supply)

    def buy(self, amount, from_token):
        """
        Exchange amount of reserve token for SmartToken
        :param amount:      amount of reserve token to be sold
        :param from_token:  name of token being sold
        :return:
        """
        assert(from_token in self.reserve_tokens.keys())
        assert(amount >= 0)
        reserve = self.reserve_tokens[from_token]
        R0 = reserve.amount
        S0 = self.supply
        self.reserve_tokens[from_token].amount += amount
        R1 = reserve.amount
        S1 = S0 * np.power(R1/R0, reserve.crr) #update supply while maintaining crr
        dS = S1 - S0
        assert(dS >= 0)
        self.supply = S1
        if self.verbose > 0:
            self.log()
            print('[BUY {n}] {amt}[{ctr}] => {dS}[{n}]'.format(
                amt=np.round(amount,3),ctr=from_token,
                dS=np.round(dS,3),n=self.name))
        return dS

    def sell(self, amount, to_token):
        """
        Exchange SmartToken for a certain amount of reserve token
        :param amount:      amount of SmartToken to be sold
        :param to_token:    name of reserve token to be bought
        :return:
        """
        assert(to_token in self.reserve_tokens.keys())
        assert(amount >= 0 and amount <= self.supply)
        reserve = self.reserve_tokens[to_token]
        R0 = reserve.amount
        S0 = self.supply
        self.supply -= amount
        S1 = self.supply
        R1 = R0 * np.power(S1/S0, 1/reserve.crr) #update reserve while maintaining crr
        self.reserve_tokens[to_token].amount = R1
        dR = R0 - R1
        assert(dR >= 0)
        if self.verbose > 0:
            self.log()
            print('[SELL {n}] {amt}[{n}] => {dR}[{ctr}] '.format(
                amt=np.round(amount,3), n=self.name,
                ctr=to_token,dR=np.round(dR,3)))
        return dR


    def exchange(self, amount, from_token, to_token):
        """
        Exchange between 2 reserve tokens using the SmartToken
        :param amount:      amount of base reserve token to be sold
        :param from_token:  name of base reserve token
        :param to_token:    name of target reserve token
        :return:
        """
        if self.verbose > 0:
            self.log()
        P0 = self.price(from_token,to_token)
        st_amount = self.buy(amount, from_token)
        to_amount = self.sell(st_amount, to_token)
        px_eff = amount/to_amount
        slip = 10000*(px_eff/P0-1)
        if self.verbose > 0:
            self.log()
            print('EXCHANGE {from_amt}[{from_token}] => {ctr_amt}[{ctr_token}] PX={px} slippage={slip}(bps)'.format(
                from_amt=np.round(amount,3),from_token=from_token,
                ctr_amt=np.round(to_amount,3),ctr_token=to_token,
                px=np.round(px_eff,4),slip=np.round(slip,3)))
        return to_amount

    def price(self, from_token, to_token):
        """
        Return the mid price for exchanging from_token => to_token
        :param from_token:
        :param to_token:
        :return:
        """
        assert (from_token in self.reserve_tokens.keys())
        assert (to_token in self.reserve_tokens.keys())
        base = self.reserve_tokens[from_token]
        ctr  = self.reserve_tokens[to_token]
        px_imp = (base.amount*ctr.crr)/(ctr.amount*base.crr )
        if self.verbose > 0:
            print('[{name}] implied {ctr}/{base}={px}'.format(
              name=self.name,ctr=to_token,base=from_token,px=np.round(px_imp,4)
            ))
        return px_imp

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

def example4(verbose):
    print('example 4 (exchange token): we show the slippage incurred by using an exchange token')
    GNO = Reserve('GNO',amount=25000,crr=0.2,verbose=1)
    GNO.log()
    ETH = Reserve('ETH',amount=100000,crr=0.8,verbose=1)
    ETH.log()
    GNOETH=SmartToken2('GNOETH',reserve_tokens=dict(),supply=1000,verbose=1)
    GNOETH.log()
    GNOETH.add_reserve(GNO)
    GNOETH.log()
    GNOETH.add_reserve(ETH)
    GNOETH.log()
    P0 = GNOETH.price('GNO','ETH')
    amt = GNOETH.exchange(10,'GNO','ETH')
    P1 = GNOETH.price('GNO','ETH')

def main():
    verbose=1
    print('='*80)
    example1(verbose)
    print('='*80)
    example2(verbose)
    print('='*80)
    example3(verbose)
    print('='*80)
    example4(verbose)

if __name__=='__main__':
    main()
