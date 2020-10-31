import abc
import numpy as np
from typing import Callable
from utils import assert_msg, SMA, crossover, kalmanF, RFfeature, save_strategy_value
from sklearn.ensemble import RandomForestRegressor
from orderbook import CrawlerToSql


class Strategy(metaclass=abc.ABCMeta):
    """
    抽象策略类， 用于定义交易策略

    如果要定义自己的策略类，需要继承这个基类，并实现两个抽象方法
    Strategy.init
    Strategy.next
    """
    def __init__(self, broker, data):
        """
        构造策略对象

        :param broker: ExchangeAPI 交易API接口，用于模拟交易
        :param data:    list        行情数据
        """
        self._indicators = []
        self._broker = broker # type: _Broker
        self._data =data #type: _Data
        self._tick = 0

    def I(self, func: Callable, *args) -> np.ndarray:
        """
        计算买卖指标向量。买卖指标向量是一个数组，长度和历史数据对应；
        用于判定这个时间点上需要进行”买“还是”卖“

        例如计算滑动平均：
        def init():
            self.sma = self.I(utils.SMA, self.data.Close, N)
        :param func:
        :param args:
        :return:
        """
        value = func(*args)
        value = np.asarray(value)
        assert_msg(value.shape[-1] == len(self._data.Close),'指示器长度必须和data长度相同')

        self._indicators.append(value)
        return value

    @property
    def tick(self):
        return self._tick

    @abc.abstractmethod
    def init(self, tick):
        """
        初始化策略。在策略回测/执行过程中调用一次，用于初始化策略内部状态
        这里也可以预计算策略的辅助参数。比如根据历史行情数据；
        计算买卖的指示器向量；
        训练模型/初始化模型参数
        :return:
        """
        pass

    @abc.abstractmethod
    def next(self, tick: int, t_stamp, *args) -> None:
        """
        步进函数，执行第tick步的策略。tick代表当前'时间'。比如data[tick]
        :param tick:
        :return:
        """
        pass

    def buy(self,*args):
        self._broker.buy(*args)

    def sell(self,*args):
        self._broker.sell(*args)

    @property
    def data(self):
        return self._data


class SmaCross(Strategy):
    def init(self, tick):
        # 小窗口SMA的窗口大小，用于计算SMA快线
        self.fast = 10

        # 大窗口SMA的窗口大小，用于计算SMA慢线
        self.slow = 20
        # 计算历史上每个时刻的快线和慢线
        self.sma1 = self.I(SMA, self.data.Close, self.fast)
        self.sma2 = self.I(SMA, self.data.Close, self.slow)

    def next(self, tick, t_stamp,*args):
        side = ''
        # 如果此时快线刚好越过慢线，买入全部
        if crossover(self.sma1[:tick], self.sma2[:tick]):
            self.buy(*args)
            side = "buy"

        # 如果是慢线刚好越过快线，卖出全部
        elif crossover(self.sma2[:tick], self.sma1[:tick]):
            self.sell(*args)
            side = "sell"

        # 否则，这个时刻不执行任何操作。
        else:

            pass

        if not self._broker._test:
            CrawlerToSql(symbol='BTCUSD', limit=5).save_to_sql(t_stamp)
            save_strategy_value(self.data.Close[tick], t_stamp, side)


class KalmanFilterPredict(Strategy):
    # 使用卡尔曼滤波通过观测值得到一个滚动的估计值
    def init(self, tick):
        self.predict = self.I(kalmanF, self.data.Close)

    def next(self, tick, t_stamp,*args):
        side = ''
        # 如果预测出来今天的股价
        if self.predict[tick] > self.predict[tick - 1]:
            self.buy(*args)
            side = "buy"

        elif self.predict[tick] < self.predict[tick - 1]:
            self.sell(*args)
            side = "sell"

        else:
            pass

        if not self._broker._test:
            CrawlerToSql(symbol='BTCUSD', limit=5).save_to_sql(t_stamp)
            save_strategy_value(self.data.Close[tick], t_stamp, side)

class RFPredcit(Strategy):
    # 使用随机森林预测涨跌
    def init(self, tick):
        low = 3
        midden = 6
        fast = 12
        self.classifier = RandomForestRegressor(n_estimators = 20)

        # 计算所需特征
        self.feature = RFfeature(self.data.Close, low, midden, fast)
        # 计算涨跌指标
        # self.changes = np.diff(self.data.Close) > 0


    def next(self, tick, t_stamp,*args):
        side = ''
        # 训练的时候使用昨天的数据预测今天的涨幅
        self.classifier.fit(self.feature[max(50, tick - 100):tick], self.data.Close[max(51, tick - 99):tick+1])

        # 预测的时候使用今天的数据预测明天的价格，如果概率大于0.5就买入
        if self.classifier.predict(self.feature[tick:tick+1])[0] > self.data.Close[tick]:
            self.buy(*args)
            side = "buy"
        # 反之就卖出
        else:
            self.sell(*args)
            side = "sell"
        if not self._broker._test:
            CrawlerToSql(symbol='BTCUSD', limit=5).save_to_sql(t_stamp)
            save_strategy_value(self.data.Close[tick], t_stamp, side)