import pandas as pd
from os import path
from pykalman import KalmanFilter

def assert_msg(condition, msg):
    if not condition:
        raise Exception(msg)

def SMA(values, n):
    """
    返回简单滑动平均
    """
    return pd.Series(values).rolling(n).mean()

def kalmanF(values):
    """
    使用卡尔曼滤波通过观测值得到一个滚动的估计值
    """

    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance = 1,
                      transition_covariance = 0.01)
    state_means, _ = kf.filter(values)
    return state_means[:,0]

def Max_retracement(value):
    """
    计算最大回撤
    :param value: list
    """
    max_top = 0
    cur_top = 0
    max_retracement = 0
    cur_retracement = 0
    for i in range(1,len(value)):
        dif = value[i] - value[i-1]
        cur_retracement -= dif
        if cur_retracement < 0:
            cur_retracement = 0
            cur_top = value[i]
        if cur_retracement > max_retracement:
            max_retracement = cur_retracement
            max_top = cur_top

    return max_retracement/max_top

def crossover(series1, series2) -> bool:
    """
    检查两个序列是否在结尾交叉
    :param series1: 序列1
    :param series2: 序列2
    :return:        如果交叉返回True, 反之返回False
    """
    return series1[-2] < series2[-2] and series1[-1] > series2[-1]

def read_file(filename):
    # 获取文件绝对路径
    filepath = path.join(path.dirname(__file__), filename)

    # 判定文件是否存在
    assert_msg(path.exists(filepath), "文件不存在")

    # 读取CSV文件并返回
    return pd.read_csv(filepath,
                       index_col=0,
                       parse_dates=True,
                       infer_datetime_format=True)

if __name__ == '__main__':
    BTCUSD = read_file('BTCUSD_GEMINI.csv')
    assert_msg(BTCUSD.__len__() > 0, '读取失败')
    print(Max_retracement([1,2,3,5,2,3,4,0,2,10,3,4,2]))
    print(BTCUSD.head())