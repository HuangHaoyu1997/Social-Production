from typing import List  
from dataclasses import dataclass  

@dataclass
class Order(object):
    Time: int # 订单的时间戳
    CreatorID: int
    
    # Side=True为offer(报价,卖方价)
    # Side=False为bids(出价,买方价)
    Side: bool
    Quantity: int   
    Price: int  

@dataclass  
class Match(object):   
    Bid: Order   
    Offer: Order     

class Market(object):
    def __init__(self):
        self.Bids: List[Order] = []
        self.Offers: List[Order] = []
        self.Matches: List[Match] = []

    def AddOrder(self, order: Order):
        if order.Side:
            self.Offers.append(order)
        else:
            self.Bids.append(order)

    def MatchOrders(self):
        # 买方价格从高到低
        self.Bids = sorted(self.Bids, key=lambda x: x.Price)[::-1] 
        # 卖方价格从低到高
        self.Offers = sorted(self.Offers, key=lambda x: x.Price)

        while (len(self.Bids) > 0 and len(self.Offers) > 0):
            # 买方最高价低于卖方最低价
            if self.Bids[0].Price < self.Offers[0].Price:
                break
            else:  # self.Bids[0].Price >= self.Offers[0].Price:
                # 取出卖方和买方队列的第1个价格
                currBid = self.Bids.pop()
                currOffer = self.Offers.pop()
                if currBid.Quantity != currOffer.Quantity: # 数量不一致
                    if currBid.Quantity > currOffer.Quantity: # 买方数量>卖方数量
                        # 买方买走卖方全部商品,将剩余订单量创建新订单,并加入Bids队列
                        newBid = Order(currBid.Time, currBid.CreatorID, currBid.Side, currBid.Quantity - currOffer.Quantity, currBid.Price)
                        self.Bids.insert(0, newBid)
                        currBid.Quantity = currOffer.Quantity
                    else:
                        newOffer = Order(currOffer.Time, currOffer.CreatorID, currOffer.Side, currOffer.Quantity - currBid.Quantity, currOffer.Price)
                        self.Offers.insert(0, newOffer)
                        currOffer.Quantity = currBid.Quantity
                # 撮合成功的订单pair
                self.Matches.append(Match(currBid, currOffer))
    
    def ClearingMatch(self):
        '''清空匹配list'''
        self.Matches: List[Match] = []
    
    def DeleteOrder(self):
        '''删除指定订单'''
        
    def ComputeClearingPrice(self) -> float:   
        '''
        计算出清价格
        '''
        if len(self.Matches) == 0:   
            return 0.
        
        # 出清价格
        clearingPrice = 0   
        # 出清数量
        cumulativeQuantity = 0
        for match in self.Matches:
            cumulativeQuantity += match.Bid.Quantity
            clearingPrice += match.Bid.Quantity * (match.Bid.Price + match.Offer.Price) / 2 # 达成交易的价格是出价和报价的均值
        # 
        return clearingPrice / cumulativeQuantity
# Create market instance and test orders   
market = Market()
buyOrder1 = Order(Time=0, CreatorID=0, Side=False, Quantity=100, Price=10.)
buyOrder2 = Order(Time=0, CreatorID=0, Side=False, Quantity=100, Price=10.)   
sellOrder1 = Order(Time=0, CreatorID=1, Side=True, Quantity=120, Price=5.5)   
sellOrder2 = Order(Time=0, CreatorID=1, Side=True, Quantity=120, Price=6)   

# Send orders to market   
market.AddOrder(buyOrder1)
market.AddOrder(buyOrder2)
market.AddOrder(sellOrder1)
market.AddOrder(sellOrder2)

# Match orders  
market.MatchOrders()

# Get the clearing price  
print(market.ComputeClearingPrice())
# returns 9  