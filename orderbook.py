import copy
import json
import ssl
import time
import websocket
import pandas as pd
import sqlite3
import config

class OrderBook(object):

    BIDS = 'bid'
    ASKS = 'ask'

    def __init__(self, limit=20):

        self.limit = limit

        # (price, amount)
        self.bids = {}
        self.asks = {}
        self.bids_sorted = {}
        self.asks_sorted = {}

    def insert(self, price, amount, direction):
        if direction == self.BIDS:
            if amount == 0:
                if price in self.bids:
                    del self.bids[price]
            else:
                self.bids[price] = amount
        elif direction == self.ASKS:
            if amount == 0:
                if price in self.asks:
                    del self.asks[price]
            else:
                self.asks[price] = amount
        else:
            print('WARNING: unknown direction {}'.format(direction))

    def sort_and_truncate(self):
        #sort
        self.bids_sorted = sorted([(price, amount) for price, amount in self.bids.items()], reverse=True)
        self.asks_sorted = sorted([(price, amount) for price, amount in self.asks.items()])

        #truncate
        self.bids_sorted = self.bids_sorted[:self.limit]
        self.asks_sorted = self.asks_sorted[:self.limit]

        #copy back to bids and asks
        self.bids = dict(self.bids_sorted)
        self.asks = dict(self.asks_sorted)

    def get_copy_of_bids_and_asks(self):
        return copy.deepcopy(self.bids_sorted), copy.deepcopy(self.asks_sorted)


class Crawler:
    def __init__(self, symbol, output_file):
        self.orderbook = OrderBook(limit=10)
        self.output_file = output_file

        self.ws = websocket.WebSocketApp('wss://api.gemini.com/v1/marketdata/{}'.format(symbol),
                                        on_message = lambda ws, message: self.on_message(message))
        self.ws.run_forever(sslopt={'cert_reqs': ssl.CERT_NONE})

    def on_message(self, message):
        #对收到的信息进行处理， 然后发送给 orderbook
        data = json.loads(message)
        for event in data['events']:
            price, amount, direction = float(event['price']), float(event['remaining']), event['side']
            self.orderbook.insert(price, amount, direction)

        #整理 orderbook，排序， 只选取我们需要的前几个
        self.orderbook.sort_and_truncate()

        #输出到文件
        with open(self.output_file, 'a+') as f:
            bids, asks = self.orderbook.get_copy_of_bids_and_asks()
            output = {
                'bids': bids,
                'asks': asks,
                'ts': int(time.time()*1000)
            }
            f.write(json.dumps(output) + '\n')


class CrawlerToSql:
    def __init__(self, symbol, limit):
        self.limit = limit
        self.orderbook = OrderBook(limit=self.limit)

        self.ws = websocket.WebSocketApp('wss://api.gemini.com/v1/marketdata/{}'.format(symbol),
                                        on_message = self.on_message)
        self.ws.run_forever(sslopt={'cert_reqs': ssl.CERT_NONE})

    def on_message(self, message):
        #对收到的信息进行处理， 然后发送给 orderbook
        data = json.loads(message)
        for event in data['events']:
            price, amount, direction = float(event['price']), float(event['remaining']), event['side']
            self.orderbook.insert(price, amount, direction)

        #整理 orderbook，排序， 只选取我们需要的前几个
        self.orderbook.sort_and_truncate()
        self.ws.close()

    def save_to_sql(self, t_stamp):

        bids, asks = self.orderbook.get_copy_of_bids_and_asks()
        bids_price = pd.DataFrame(dict(zip([str(i) + '_bid_price' for i in range(1, self.limit + 1)],
                              [bids[i][0] for i in range(0, self.limit)])), index=[t_stamp])
        bids_amount = pd.DataFrame(dict(zip([str(i) + '_bid_amount' for i in range(1, self.limit + 1)],
                              [bids[i][1] for i in range(0, self.limit)])), index=[t_stamp])
        asks_price = pd.DataFrame(dict(zip([str(i) + '_ask_price' for i in range(1, self.limit + 1)],
                              [bids[i][0] for i in range(0, self.limit)])), index=[t_stamp])
        asks_amount = pd.DataFrame(dict(zip([str(i) + '_ask_amount' for i in range(1, self.limit + 1)],
                              [bids[i][1] for i in range(0, self.limit)])), index=[t_stamp])

        conn = sqlite3.connect('D:\\try\\quant_start2\\test.db')
        bids_price.to_sql("bids_price", conn, if_exists='append')
        bids_amount.to_sql("bids_amount", conn, if_exists='append')
        asks_price.to_sql("asks_price", conn, if_exists='append')
        asks_amount.to_sql("asks_amount", conn, if_exists='append')
        conn.close()

if __name__ == '__main__':
    #crawler = Crawler(symbol='BTCUSD', output_file='BTCUSD.txt')
    CrawlerToSql(symbol='BTCUSD',limit = 5).save_to_sql(t_stamp)