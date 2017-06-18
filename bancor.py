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
    def __init__(self, name, supply, reserve_tokens=dict(), verbose=0, precision=4):
        self.name = name
        self.reserve_tokens = reserve_tokens
        self.supply = supply
        self.verbose=verbose
        self.precision=precision
        if self.verbose > 0:
            self.log()

    def log(self):
        if self.verbose > 0:
            print('[{}][LOG]'.format(self.name)
                  + ' S={s} CRR={crr}:'.format(crr=np.round(self.crr, self.precision), s=np.round(self.supply, self.precision)) \
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
        return np.sum([r.crr for _, r in self.reserve_tokens.items()])

    @property
    def reserve(self):
        reserve_tot = np.sum([r.amount for _,r in self.reserve_tokens.items()])
        assert(reserve_tot >= 0)
        return reserve_tot

    @property
    def price(self):
        return self.reserve/(self.crr * self.supply)

    def partial_prices(self):
        return dict(zip(self.reserve_tokens.keys(),
                        [self.__partial_price(tkn) for tkn in self.reserve_tokens.keys()]))

    def __partial_price(self, token, reserve_amount=None, supply=None):
        """
        Implied price of the SmartToken if one buys it using the given token
        :param token:
        :param reserve_amount: if unset used the actual reserve amount otherwise this price is sim
        :return:
        """
        assert(token in self.reserve_tokens.keys())
        reserve = self.reserve_tokens[token]
        assert (type(reserve) is Reserve)
        if reserve_amount is None:
            reserve_amount = reserve.amount
        if supply is None:
            supply = self.supply
        return reserve_amount / (reserve.crr * supply)

    def buy(self, amount, from_token, const=False):
        """
        Exchange amount of reserve token for SmartToken
        :param amount:      amount of reserve token to be sold
        :param from_token:  name of token being sold
        :param const: set to True to get the answer without modifying the object
        :return:
        """
        assert(from_token in self.reserve_tokens.keys())
        assert(amount >= 0)
        reserve = self.reserve_tokens[from_token]
        P0 = self.__partial_price(from_token)
        R0 = reserve.amount
        S0 = self.supply
        R1 = R0 + amount
        if not const:
            self.reserve_tokens[from_token].amount = R1
        S1 = S0 * np.power(R1/R0, reserve.crr) #update supply while maintaining crr
        dS = S1 - S0
        assert(dS >= 0)
        if not const:
            self.supply = S1
        if self.verbose > 1:
            self.log()
        if self.verbose > 0:
            px_eff = amount / dS
            P1 = self.__partial_price(from_token, R1, S1)
            slippage = 10000 * (px_eff / P0 - 1)
            print('[{n}][BUY] {amt}[{ctr}] => {dS}[{n}] ; P0={P0} P1={P1} PAY={px} SLIP={slip}(bps)'.format(
                amt=np.round(amount, self.precision), ctr=from_token,
                dS=np.round(dS, self.precision), n=self.name,
                P0=np.round(P0, self.precision), P1=np.round(P1, self.precision),
                px=np.round(px_eff, self.precision), slip=np.round(slippage, self.precision)))
        return dS

    def sell(self, amount, to_token, const=False):
        """
        Exchange SmartToken for a certain amount of reserve token
        :param amount:      amount of SmartToken to be sold
        :param to_token:    name of reserve token to be bought
        :param const: set to True to get the answer without modifying the object
        :return:
        """
        assert(to_token in self.reserve_tokens.keys())
        assert(amount >= 0 and amount <= self.supply)
        reserve = self.reserve_tokens[to_token]
        P0 = self.__partial_price(to_token)
        R0 = reserve.amount
        S0 = self.supply
        S1 = S0 - amount
        if not const:
            self.supply = S1

        R1 = R0 * np.power(S1/S0, 1/reserve.crr) #update reserve while maintaining crr
        if not const:
            self.reserve_tokens[to_token].amount = R1
        dR = R0 - R1
        assert(dR >= 0)
        if self.verbose > 1:
            self.log()
        if self.verbose > 0:
            px_eff = dR/amount
            slippage = 10000*(1-px_eff/P0)
            P1 = self.__partial_price(to_token, R1, S1)
            print('[{n}][SELL] {amt}[{n}] => {dR}[{ctr}] ; P0={P0} P1={P1} REC={px} SLIP={slip}(bps)'.format(
                amt=np.round(amount, self.precision), n=self.name,
                ctr=to_token, dR=np.round(dR, self.precision),
                P0=np.round(P0, self.precision), P1=np.round(P1, self.precision),
                px=np.round(px_eff, self.precision), slip=np.round(slippage, self.precision)
            ))
        return dR

    def exchange(self, amount, from_token, to_token, const=False):
        """
        Exchange between 2 reserve tokens using the SmartToken
        :param amount:      amount of base reserve token to be sold
        :param from_token:  name of base reserve token
        :param to_token:    name of target reserve token
        :param const: set to True to get the answer without modifying the object
        :return:
        """
        if self.verbose > 0:
            self.log()
        P0 = self.exchange_price(from_token,to_token)
        st_amount = self.buy(amount, from_token, const)
        to_amount = self.sell(st_amount, to_token, const)
        px_eff = amount/to_amount
        slip = 10000*(px_eff/P0-1)
        if self.verbose > 1:
            self.log()
        if self.verbose > 0:
            print('[{n}][EXCH] {from_amt}[{from_token}] => {ctr_amt}[{ctr_token}] PAY={px} SLIP={slip}(bps)'.format(
                n=self.name, from_amt=np.round(amount, self.precision),from_token=from_token,
                ctr_amt=np.round(to_amount, self.precision),ctr_token=to_token,
                px=np.round(px_eff, self.precision), slip=np.round(slip, self.precision)))
        return to_amount

    def exchange_price(self, from_token, to_token):
        """
        Return the mid price for exchanging from_token => to_token
        :param from_token:
        :param to_token:
        :return:
        """
        if from_token == self.name:
            return self.__partial_price(to_token)
        elif to_token == self.name:
            return 1/self.__partial_price(from_token)

        assert (from_token in self.reserve_tokens.keys())
        assert (to_token in self.reserve_tokens.keys())
        base = self.reserve_tokens[from_token]
        ctr = self.reserve_tokens[to_token]
        px_imp = (base.amount*ctr.crr)/(ctr.amount*base.crr)
        if self.verbose > 1:
            self.log()
        if self.verbose > 0:
            print('[{n}][IMP] {ctr}/{base}={px}'.format(
              n=self.name, ctr=to_token, base=from_token, px=np.round(px_imp, self.precision)
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
    P0 = GNOETH.exchange_price('GNO', 'ETH')
    amt = GNOETH.exchange(10,'GNO', 'ETH')
    P1 = GNOETH.exchange_price('GNO', 'ETH')

def example5(verbose):
    print('where we show that for a basket token there is a cheapest reserve to buy the smart token')
    GNO = Reserve('GNO',amount=25000, crr=0.2, verbose=1)
    ETH = Reserve('ETH',amount=100000, crr=0.8, verbose=1)
    GNOETH=SmartToken2('GNOETH', reserve_tokens=dict(GNO=GNO, ETH=ETH), supply=1000, verbose=1, precision=4)
    print('all partial prices are equal:{}'.format(GNOETH.partial_prices()))
    P0 = GNOETH.exchange_price('GNO','ETH')
    amt = GNOETH.exchange(100,'GNO','ETH')
    P1 = GNOETH.exchange_price('GNO','ETH')
    print('Displayed SmartToken price: {}'.format(np.round(GNOETH.price, 4)))
    print('now partial prices are different, SmartToken is cheaper in ETH!: {}'.format(GNOETH.partial_prices()))
    print('proof:')
    GNOETH.buy(100, 'ETH', const=True)
    GNOETH.buy(100, 'GNO', const=True)

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
    print('='*80)
    example5(verbose)

if __name__=='__main__':
    main()
