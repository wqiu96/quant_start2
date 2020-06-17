import requests
import pandas as pd


# 获取Gemini交易所报价数据
def get_price():
    gemini_ticker = "https://api.gemini.com/v1/pubticker/{}"
    symbol = "btcusd"
    btc_data = requests.get(gemini_ticker.format(symbol)).json()
    return float(btc_data['last'])

# 获取最近一个小时交易数据并绘图
def get_last1000_min_price():
    # 要获取的数据时间段
    periods = "60"

    # 抓取数据
    resp = requests.get("https://api.cryptowat.ch/markets/gemini/btcusd/ohlc", params={
        "periods": periods
    })
    data = resp.json()
    # 转换成pandas的data frame
    df = pd.DataFrame(
        data["result"][periods],
        columns=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "NA"
        ]
    )

    # 输出

    return df

if __name__ == "__main__":
    get_price()
    res = get_last1000_min_price()