import numpy as np
import pandas as pd
from utils import assert_msg, read_file, Max_retracement
from matplotlib import pyplot as plt
from strategy import Strategy, SmaCross, KalmanFilterPredict, RFPredcit

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
        self._test = True

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


        assert_msg(issubclass(strategy_type, Strategy),'strategy_type不是一个Strategy类型')
        assert_msg(issubclass(broker_type, ExchangeAPI), 'broker_type不是一个ExchangeAPI类型')
        assert_msg(isinstance(commission, float), 'commission不是浮点数值类型')

        self._strategy_value = []
        self._strategy_return = []

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
        s['Initial value'] = broker.initial_cash
        s['Closing value'] = broker.market_value
        s['Return'] = broker.market_value - broker.initial_cash
        s['Maximum retracement rate'] = Max_retracement(self._strategy_value)
        s['Return variance'] = np.std(self._strategy_return)
        return s

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
