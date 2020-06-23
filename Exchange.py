import requests
import json
import base64
import hmac
import hashlib
import datetime, time
import numpy as np
import pandas as pd
from utils import assert_msg, Max_retracement, plotres
from Get_Price import get_price, get_last1000_min_price
from strategy import Strategy, SmaCross, KalmanFilterPredict, RFPredcit
import sqlite3


class RealExchangeAPI:
    def __init__(self):
        self._base_url = "https://api.sandbox.gemini.com"
        self._exendpoint = "/v1/order/new"
        self._balanceend = "/v1/balances"

        self.__gemini_api_key = "account-lDs0Oamjmhfke0Cd63Sw"
        self.__gemini_api_secret = "CqLUV7NqU9yqiSqLxX8fZNZrktE".encode()
        self._inital_btc= self.__balance('BTC')
        self._inital_cash = self.__balance('USD')
        self._test = False

    def __exchange(self, symbol, amount, price, side, Extype):
        """
        :param symbol: string   Example: 'btcusd"
        :param amount: string   Example: '1'
        :param price:  string   Example: '9400'
        :param side:   string   Example: 'buy' or 'side'
        :param type:   string   Example: 'exchange limit'
        :return:
        """
        assert_msg(type(symbol) == str, 'symbol不是一个str类型')
        assert_msg(type(amount) == str, 'amount不是一个str类型')
        assert_msg(type(price) == str, 'price不是一个str类型')
        assert_msg(type(side) == str, 'side不是一个str类型')
        assert_msg(type(Extype) == str, 'Extype不是一个str类型')

        t = datetime.datetime.now()
        payload_nonce = str(int(time.mktime(t.timetuple())))
        time.sleep(1)

        url = self._base_url + self._exendpoint
        payload = {"request": "/v1/order/new",
                   "nonce": payload_nonce,
                   "symbol": symbol,
                   "amount": amount,
                   "price": price,
                   "side": side,
                   "type": Extype,
                   }

        encoded_payload = json.dumps(payload).encode()
        b64 = base64.b64encode(encoded_payload)
        signature = hmac.new(self.__gemini_api_secret, b64, hashlib.sha384).hexdigest()

        request_headers = {'Content-Type': "text/plain",
                           'Content-Length': "0",
                           'X-GEMINI-APIKEY': self.__gemini_api_key,
                           'X-GEMINI-PAYLOAD': b64,
                           'X-GEMINI-SIGNATURE': signature,
                           'Cache-Control': "no-cache"}

        response = requests.post(url,
                                 data=None,
                                 headers=request_headers)

        #assert_msg(type(response) == list, response)
        new_clearing_order = response.json()
        #print(new_clearing_order)

    def __balance(self, symbol ,*args):
        """
        :param symbol: Example:"USD","BTC"
        :param args: subamount(optional)
        :return:
        """
        assert_msg(type(symbol) == str, 'symbol不是一个str类型')
        t = datetime.datetime.now()
        payload_nonce = str(int(time.mktime(t.timetuple())))
        time.sleep(1)

        url = self._base_url + self._balanceend
        payload = {"request": "/v1/balances",
                   "nonce": payload_nonce,
                   }
        encoded_payload = json.dumps(payload).encode()
        b64 = base64.b64encode(encoded_payload)
        signature = hmac.new(self.__gemini_api_secret, b64, hashlib.sha384).hexdigest()

        request_headers = {'Content-Type': "text/plain",
                           'Content-Length': "0",
                           'X-GEMINI-APIKEY': self.__gemini_api_key,
                           'X-GEMINI-PAYLOAD': b64,
                           'X-GEMINI-SIGNATURE': signature,
                           'Cache-Control': "no-cache"}

        response = requests.post(url,
                                 data=None,
                                 headers=request_headers)

        balance = response.json()
        #assert_msg(type(balance) == list, balance)
        res = list(filter(lambda item: item if item['currency'] == symbol else None, balance))
        return float(res[0]['available'])

    @property
    def cash(self):
        """
        :return: 返回当前账号现金数量
        """
        return self.__balance('USD')

    @property
    def position(self):
        """
        :return:返回当前账号仓位
        """
        return self.__balance('BTC')

    @property
    def initial_cash(self):
        """
        :return: 返回初始现金数量
        """
        return self._inital_cash

    @property
    def initial_btc(self):
        """
        :return: 返回初始比特币数量
        """
        return self._inital_btc

    @property
    def market_value(self):
        """
        :return:返回当前市值
        """
        return self.cash + self.position*get_price()

    @property
    def current_price(self):
        """
        :return:返回当前市场价格
        """
        return get_price()

    def buy(self, symbol, amount, Extype):

        self.__exchange(symbol, amount, "10000", "buy", Extype)

    def sell(self, symbol, amount, Extype):

        self.__exchange(symbol, amount, "200", "sell", Extype)

    def next(self, tick):
        self._i = tick


class Autoexchange:

    """
   Autoexchange 自动交易类，用于读取当前行情、执行策略、交易并计算收益。

    初始化的时候调用Autoexchange.run来交易

    """

    def __init__(self,
                 strategy_type: type(Strategy),
                 broker_type: type(RealExchangeAPI),
                 ):

        assert_msg(issubclass(strategy_type, Strategy),'strategy_type不是一个Strategy类型')
        assert_msg(issubclass(broker_type, RealExchangeAPI), 'strategy_type不是一个Real ExchangeAPI类型')

        self._strategy_value = []
        self._strategy_return = []

        self._broker = broker_type()
        self._data = get_last1000_min_price()
        self._strategy = strategy_type
        self._results = None

    def run(self, Escape_time, *args):
        """
        :param Escape_time: value:second type int
        :return:
        """
        broker = self._broker
        tick = len(self._data) - 1 #In real excahnge, start time is always NOW
        broker.next(tick)  #In real excahnge, start time is always NOW

        self._init_value = broker.market_value

        t_star = datetime.datetime.now()
        t_end = datetime.datetime.now()

        while (t_end -  t_star).seconds <= Escape_time:

            self._data = get_last1000_min_price() #updata the data
            strategy = self._strategy(self._broker, self._data)
            strategy.init(tick)

            self._strategy_value.append(broker.market_value) # 记录每时每刻的市值

            #把每时每刻的市值记录到数据库
            conn = sqlite3.connect('D:\\try\\quant_start2\\test.db')
            df = pd.DataFrame({"Market_value": self._strategy_value[-1]}, index=[0])
            df.to_sql("Market_value", conn, if_exists='append')
            conn.close()

            strategy.next(tick, *args)
            time.sleep(60) #rest and wait for data updata
            t_end = datetime.datetime.now()


        # 计算收益率变化
        self._strategy_return = (np.array(np.array(self._strategy_value[1:]) - np.array(self._strategy_value[:-1]))
                                /np.array(self._strategy_value[:-1]))

        # 完成策略执行之后，计算结果并返回
        self._results = self._compute_result(broker)
        return self._results, self._strategy_value

    def _compute_result(self, broker):
        s = pd.Series()
        s['Initial value'] = self._init_value
        s['Closing value'] = broker.market_value
        s['Return'] = broker.market_value - self._init_value
        s['Maximum retracement rate'] = Max_retracement(self._strategy_value)
        s['Return variance'] = np.std(self._strategy_return)
        s = s.round(2)
        return s


def main():
    res, strategy_value = Autoexchange(KalmanFilterPredict, RealExchangeAPI).run(120,"btcusd", "1", "exchange limit")
    print(res)
    print(strategy_value[-1])
    plotres(res, strategy_value)


if __name__ == '__main__':
    main()

