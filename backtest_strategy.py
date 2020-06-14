import abc
import numpy as np
import pandas as pd
from typing import Callable
from utils import assert_msg, SMA, crossover, read_file, kalmanF, Max_retracement
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from collections import deque

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
    def next(self, tick):
        """
        步进函数，执行第tick步的策略。tick代表当前'时间'。比如data[tick]
        :param tick:
        :return:
        """
        pass

    def buy(self):
        self._broker.buy()

    def sell(self):
        self._broker.sell()

    @property
    def data(self):
        return self._data

class ExchangeAPI:
    def __init__(self, data, cash, commission):
        assert_msg(0 < cash, "初始现金数量大于0，输入的现金数量：{}".format(cash))
        assert_msg(0 <= commission <= 0.05, "合理的手续费率一般不会超过5%，输入的费率：{}".format(commission))
        self._inital_cash = cash
        self._data = data
        self._commission = commission
        self._position = 0
        self._cash = cash
        self._i = 0

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
        return self._position

    @property
    def initial_cash(self):
        """
        :return: 返回初始现金数量
        """
        return self._inital_cash

    @property
    def market_value(self):
        """
        :return:返回当前市值
        """
        return self._cash + self._position*self.current_price

    @property
    def current_price(self):
        """
        :return:返回当前市场价格
        """
        return self._data.Close[self._i]

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

class Backtest:
    """
    Backtest 回测类，用于读取历史行情、执行策略、模拟交易并估计收益。

    初始化的时候调用Backtest.run来回测

    instance, or 'backtesting.backtesting.Backtest.optimize' to optimize it
    """

    def __init__(self,
                 data: pd.DataFrame,
                 strategy_type: type(Strategy),
                 broker_type: type(ExchangeAPI),
                 cash: float = 10000,
                 commission: float = .0):
        """
        构造回测对象。需要的参数包括：历史数据，策略对象，初始资金数量，手续费率等。
        初始化过程包括检测输入类型，填充数据空值等。

        参数：
        ：param data:            pd.DataFrame            pandas Dataframe格式的历史OHLCV数据
        :param broker_type:     type(ExchangeAPI)       交易所API类型，负责执行买卖操作以及账户状态的维护
        :param strategy_type:   type(Strategy)          策略类型
        :param cash:            float                   初始资金数量
        :param commission:      float                   每次交易手续费率。如2%的手续费此处为0.02
        """

        self._strategy_value = []
        self._strategy_return = []
        assert_msg(issubclass(strategy_type, Strategy),'strategy_type不是一个Strategy类型')
        assert_msg(issubclass(broker_type, ExchangeAPI), 'strategy_type不是一个Strategy类型')
        assert_msg(isinstance(commission, float), 'commission不是浮点数值类型')

        data = data.copy(False)

        # 如果没有Volumn列，填充NaN
        if 'Volume' not in data:
            data['Volume'] = np.nan

        # 验证OHLC数据格式
        assert_msg(len(data.columns & {'Open', 'High', 'Low', 'Close', 'Volume'}) == 5,
                   ("输入的`data`格式不正确，至少需要包含这些列："
                    "" "'Open', 'High', 'Low', 'Close'"))

        # 检查缺失值
        assert_msg(not data[['Open', 'High', 'Low', 'Close']].max().isnull().any(),
                   ('部分OHLC包含缺失值，请去掉那些行或者通过差值填充. '))

        # 如果行情数据没有按照时间排序，重新排序一下
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        # 利用数据，初始化交易所对象和策略对象。
        self._data = data # type: pd.DataFrame
        self._broker = broker_type(data, cash, commission)
        self._strategy = strategy_type(self._broker, self._data)
        self._results = None

    def run(self):
        """ 运行回测，迭代历史数据，执行模拟交易并返回回测结果。
        Run the backtest. Returns `pd.Series` with results and statistics.

        Keyword arguments are interpreted as strategy parameters.
        """
        strategy = self._strategy
        broker = self._broker


        # 设定回测开始和结束位置
        start = 100
        # 策略初始化
        strategy.init(start)
        end = len(self._data)

        # 回测主循环，更新市场状态，然后执行策略
        for i in range(start, end):
            # 注意要先把市场状态移动到第i时刻，然后再执行策略。
            broker.next(i)
            # 记录每时每刻的市值
            self._strategy_value.append(broker.market_value)
            strategy.next(i)

        # 计算收益率变化
        self._strategy_return = (np.array(np.array(self._strategy_value[1:]) - np.array(self._strategy_value[:-1]))
                                /np.array(self._strategy_value[:-1]))

        # 完成策略执行之后，计算结果并返回
        self._results = self._compute_result(broker)
        return self._results, self._strategy_value, self._strategy_return

    def _compute_result(self, broker):
        s = pd.Series()
        s['初始市值'] = broker.initial_cash
        s['结束市值'] = broker.market_value
        s['收益'] = broker.market_value - broker.initial_cash
        s['最大回撤率'] = Max_retracement(self._strategy_value)
        s['收益波动率'] = np.std(self._strategy_return)
        return s

class SmaCross(Strategy):
    # 小窗口SMA的窗口大小，用于计算SMA快线
    fast = 10

    # 大窗口SMA的窗口大小，用于计算SMA慢线
    slow = 20

    def init(self, tick):
        # 计算历史上每个时刻的快线和慢线
        self.sma1 = self.I(SMA, self.data.Close, self.fast)
        self.sma2 = self.I(SMA, self.data.Close, self.slow)

    def next(self, tick):
        # 如果此时快线刚好越过慢线，买入全部
        if crossover(self.sma1[:tick], self.sma2[:tick]):
            self.buy()

        # 如果是慢线刚好越过快线，卖出全部
        elif crossover(self.sma2[:tick], self.sma1[:tick]):
            self.sell()

        # 否则，这个时刻不执行任何操作。
        else:
            pass

class KalmanFilterPredict(Strategy):
    #使用卡尔曼滤波通过观测值得到一个滚动的估计值
    def init(self, tick):
        self.predict = self.I(kalmanF, self.data.Close)

    def next(self, tick):
        # 如果预测出来今天的股价
        if self.predict[tick] > self.predict[tick - 1]:
            self.buy()

        elif self.predict[tick] < self.predict[tick - 1]:
            self.sell()

        else:
            pass

class RFPredcit(Strategy):
    # 使用随机森林预测涨跌
    def init(self, tick):
        window_length = 10
        self.classifier = RandomForestClassifier(n_estimators = 10)

        # 先进先出的deque序列，设定了最长的长度，在序列超过最长长度的时候，会将头部序列移出
        self._up_down = deque(maxlen = 100) #记录涨跌
        self._price_list = deque(maxlen = 100) #记录价格序列
        self._recent_prices = deque(maxlen = window_length + 2)  # 保存最近的股价

        #初始化队列
        for i in range(window_length + 2):
            self._recent_prices.append(self.data.Close[i])

        for i in range(window_length + 2, tick):
            changes = np.diff(self._recent_prices) > 0
            self._price_list.append(list(self._recent_prices)[0:-1])
            self._up_down.append(changes[-1])

    def next(self, tick):
        self._recent_prices.append(tick)
        self.classifier.fit(self._price_list, self._up_down)

        #如果概率大于0.5就买入
        if self.classifier.predict(np.array(self._recent_prices)[0:-1].reshape(1,-1)) > 0.5:
            self.buy()

        # 反之就卖出
        else:
            self.sell()

        self._price_list.append(list(self._recent_prices)[0:-1])
        changes = np.diff(self._recent_prices) > 0
        self._up_down.append(changes[-1])




def main():
    BTCUSD = read_file('BTCUSD_GEMINI.csv')
    ret, strategy_value, strategy_return = Backtest(BTCUSD[10000:], RFPredcit, ExchangeAPI, 10000.0, 0.00).run()
    print(ret)
    print(strategy_value[-1])
    plt.xlabel('Time')
    plt.ylabel('value')

    x = [i for i in range(len(strategy_value))]
    plt.plot(x, strategy_value, label = 'strategy Value')
    plt.plot(x, (10000/BTCUSD.Close[10100])*BTCUSD.Close[10100:], label = 'BTCUSD Value')

    plt.grid(True)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
