import requests
import json
import base64
import hmac
import hashlib
import datetime, time
import numpy as np
import pandas as pd
from utils import assert_msg, read_file, Max_retracement
from Get_Price import get_price

class Real_ExchangeAPI:
    def __init__(self, init_btc, init_cash):
        assert_msg(0 < init_btc, "初始比特币数量大于0，输入的现金数量：{}".format(init_btc))
        assert_msg(0 < init_cash, "初始现金数量大于0，输入的现金数量：{}".format(cash))
        self._inital_btc= init_btc
        self._inital_cash = init_cash
        self._btc = init_btc
        self._cash = init_cash

        self._base_url = "https://api.sandbox.gemini.com"
        self._endpoint = "/v1/order/new"
        self._url = base_url + endpoint

        self.__gemini_api_key = "account-lDs0Oamjmhfke0Cd63Sw"
        self.__gemini_api_secret = "CqLUV7NqU9yqiSqLxX8fZNZrktE".encode()


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

        response = requests.post(self._url,
                                 data=None,
                                 headers=request_headers)

        new_clearing_order = response.json()
        print(new_clearing_order)

    @property
    def cash(self):
        """
        :return: 返回当前账号现金数量
        """
        return self._cash

    @property
    def position(self):
        """
        :return:返回当前账号仓位
        """
        return self._btc

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
        return self._cash + self._btc*get_price()

    @property
    def current_price(self):
        """
        :return:返回当前市场价格
        """
        return get_price()

    def buy(self):
        """
        用当前账户剩余资金，按照市场价格全部买入
        """
        self._position += float(self._cash / (self.current_price * (1 + self._commission)))
        self._cash = 0.0

    def sell(self):
        """
        卖出当前账户剩余持仓
        """
        self._cash += float(self._position * self.current_price * (1 - self._commission))
        self._position = 0.0

    def next(self, tick):
        self._i = tick





