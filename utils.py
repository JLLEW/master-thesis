import pandas as pd
import numpy as np
import yfinance as yf
import random


def process(path):
    print(path)
    df = pd.read_csv(path, sep=',')
    #df['open_time'] = pd.to_datetime(df['open_time'],unit='ms')
    #df['close_time'] = pd.to_datetime(df['close_time'],unit='ms')
    df = df.drop(columns=['open_time', 'close_time', 'quote_volume', 'taker_buy_volume', 'count', 'taker_buy_quote_volume', 'ignore'])

    return df

def load_data(asset_list):
    print('Loading data...')
    assets_dict = {}
    for asset in asset_list:
        path = paths[asset]
        assets_dict[asset] = process(path)

    return assets_dict

def download_data(tickers, start, end, interval):
    data = yf.download(tickers=tickers, start = start, end = end, interval=interval)
    return data

def yfinance_data_loader(asset_list, start, end, interval):
    print('Downloading data...')
    assets_dict = {}
    for asset in asset_list:
        assets_dict[asset] = download_data(asset, start, end, interval)

    return assets_dict

def get_covariance_matrix(data):
    df = pd.DataFrame(data)
    #returns = np.log(df).diff()
    returns = df.pct_change().dropna()
    cov_matrix = returns.cov()

    return cov_matrix


def data_generator(steps, probs, start, low, high):
    data = []

    val = start
    for _ in range(steps):
        r = random.random()
        # data value drops
        if r < probs[0]:
            diff = random.uniform(low, 0)
        # data value goes up
        else:
            diff = random.uniform(0, high)

        val += diff
        data.append(val)

    return data

def generate_train_assets():
    stock1 = data_generator(100, [0.2, 0.8], 100, 2, 0)
    periods = 6
    for _ in range(periods):
        stock1 += data_generator(100, [0.8, 0.2], stock1[-1], 0, -1)
        stock1 += data_generator(100, [0.2, 0.8], stock1[-1], 2, 0)
    stock1 += data_generator(100, [0.8, 0.2], stock1[-1], 0, -1)

    stock2 = data_generator(1400, [0.5, 0.5], 560, 1, -1)
    stock3 = data_generator(1400, [0.3, 0.7], 10, 2, 0)
    stock4 = data_generator(1400, [0.7, 0.3], 4821, 0, -2)

    assets = {
        'W_stock': stock1,
        'S_stock': stock2,
        'Bull_stock': stock3,
        'Bear_stock': stock4
    }
    
    return assets

def generate_test_assets():
    stock1 = data_generator(100, [0.2, 0.8], 86, 3, 0)
    stock1 += data_generator(100, [0.7, 0.3], stock1[-1], 0, -2)
    stock1 += data_generator(100, [0.2, 0.8], stock1[-1], 3, 0)
    stock1 += data_generator(100, [0.77, 0.23], stock1[-1], 0, -2)

    stock2 = data_generator(400, [0.5, 0.5], 230, 2, -2)
    stock3 = data_generator(400, [0.3, 0.7], 59, 2, 0)
    stock4 = data_generator(400, [0.7, 0.3], 1230, 0, -2)

    assets = {
        'W_stock': stock1,
        'S_stock': stock2,
        'Bull_stock': stock3,
        'Bear_stock': stock4
    }
    
    return assets

def generate_test_data(scenario):
    if scenario == '3bears_1bull':
        assets = generate_3bears_1bull()
    elif scenario == 'mix':
        assets = generate_mix()
    elif scenario == 'mix_new':
        _, assets = generate_data_mix_new()
    elif scenario == '1bear_1bull':
        assets = generate_1bear_1bull()
    elif scenario =='2bear_1bull':
        assets = generate_2bear_1bull()
    elif scenario == 'single_w':
        assets = generate_w_stock()
    else:
        assets = generate_test_assets()
    output_dict = {}
    for asset in assets.keys():
        output_dict[asset] = pd.DataFrame({'price': assets[asset]})

    return output_dict, get_assets_name_list(scenario)

def generate_train_data(scenario):
    if scenario == '3bears_1bull':
        assets = generate_3bears_1bull()
    elif scenario == 'mix':
        assets = generate_mix()
    elif scenario == 'mix_new':
        assets, _ = generate_data_mix_new()
    elif scenario == '1bear_1bull':
        assets = generate_1bear_1bull()
    elif scenario =='2bear_1bull':
        assets = generate_2bear_1bull()
    elif scenario == 'single_w':
        assets = generate_w_stock()
    else:
        assets = generate_train_assets()
    output_dict = {}
    for asset in assets.keys():
        output_dict[asset] = pd.DataFrame({'price': assets[asset]})

    return output_dict, get_assets_name_list(scenario)

def generate_w_stock():
    start_price = random.randint(120, 150)
    stock1 = data_generator(50, [0.2, 0.8], start_price, 0, 2)
    periods = 20
    for _ in range(periods):
        stock1 += data_generator(50, [0.8, 0.2], stock1[-1], -2, 0)
        stock1 += data_generator(50, [0.2, 0.8], stock1[-1], 0, 2)
    stock1 += data_generator(50, [0.8, 0.2], stock1[-1], -2, 0)


    assets = {
        'w_stock': stock1
    }

    return assets

def generate_3bears_1bull():
    stock1_bear = data_generator(600, [0.7, 0.3], 720, -1, 0.2)
    stock2_bear = data_generator(600, [0.6, 0.4], 3500, -3, 0.7)
    stock3_bear = data_generator(600, [0.95, 0.05], 800, -0.5, 0.1)
    stock4_bull = data_generator(600, [0.1, 0.9], 120, -0.2, 1.5)

    assets = {
        'Bear_stock_1': stock1_bear,
        'Bear_stock_2': stock2_bear,
        'Bear_stock_3': stock3_bear,
        'Bull_stock_4': stock4_bull
    }

    return assets

def generate_1bear_1bull():
    stock1_bull = data_generator(1200, [0.2, 0.8], random.randint(10, 100), -0.2, random.randint(1, 3))
    stock2_bear = data_generator(1200, [0.8, 0.2], random.randint(12000, 25000), -10, random.randint(1, 2))

    assets = {
        'stock1_bull': stock1_bull,
        'stock2_bear': stock2_bear
    }

    return assets

def generate_2bear_1bull():
    stock1_bull = data_generator(1200, [0.2, 0.8], random.randint(10, 100), -0.2, random.randint(1, 3))
    stock2_bear = data_generator(1200, [0.8, 0.2], random.randint(12000, 25000), -10, random.randint(1, 2))
    stock3_bear = data_generator(5000, [0.7, 0.3], random.randint(12000, 25000), -13, random.randint(1, 2))
    assets = {
        'stock1_bull': stock1_bull,
        'stock2_bear': stock2_bear,
        'stock3_bear': stock3_bear
    }

    return assets

def generate_mix():
    bull_high_vol = data_generator(3000, [0.45, 0.55], random.randint(4700, 6000), -3, 5)
    bull_low_vol = data_generator(3000, [0.3, 0.7], random.randint(5100, 5300), -1.5, 1.5)
    bull_very_low_vol = data_generator(3000, [0.43, 0.57], random.randint(1500, 2000), -0.1, 0.3)
    bear = data_generator(3000, [0.6, 0.4], random.randint(12000, 14000), -3, 2.5)
    side = data_generator(3000, [0.5, 0.5], random.randint(1800, 2000), -1, 1)
    side_high_vol = data_generator(3000, [0.5, 0.5], random.randint(15000, 16000), -5, 5)
    
    assets = {
        'bull_h_vol': bull_high_vol,
        'bull_l_vol': bull_low_vol,
        'bull_vl_vol': bull_very_low_vol,
        'bear': bear,
        'side': side,
        'side_h_vol': side_high_vol
    }

    return assets

def train_test_split(stocks, train_ratio=0.7):
    train_data = {}
    test_data = {}
    validation_data = {}
    
    for stock_name, df in stocks.items():
        first_split_idx = int(len(df) * train_ratio)
        second_split_idx = int(len(df) * (train_ratio + 0.15))
        train_data[stock_name] = df.iloc[:first_split_idx]
        test_data[stock_name] = df.iloc[first_split_idx:second_split_idx]
        validation_data[stock_name] = df.iloc[second_split_idx:]
    
    return train_data, test_data, validation_data

def train_test_split2(stocks, train_ratio=0.7):
    train_data = {}
    test_data = {}
    
    for stock_name, df in stocks.items():
        first_split_idx = int(len(df) * train_ratio)
        train_data[stock_name] = df.iloc[:first_split_idx]
        test_data[stock_name] = df.iloc[first_split_idx:]
    
    return train_data, test_data

def generate_artificial_stock_data(n_days=252*12):
    
    def generate_bullish_stock(n_days, base_price, annual_growth_rate):
        daily_growth_rate = (1 + annual_growth_rate) ** (1/252) - 1
        return base_price * np.cumprod(1 + np.random.normal(loc=daily_growth_rate, scale=0.01, size=n_days))

    def generate_bearish_stock(n_days, base_price, annual_decline_rate):
        daily_decline_rate = (1 + annual_decline_rate) ** (1/252) - 1
        return base_price * np.cumprod(1 + np.random.normal(loc=daily_decline_rate, scale=0.02, size=n_days))

    def generate_sideways_stock(n_days, base_price, high_volatility=False):
        scale = 0.04 if high_volatility else 0.015
        return base_price * np.cumprod(1 + np.random.normal(loc=0, scale=scale, size=n_days))

    def generate_cyclical_stock(n_days, base_price):
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        price = base_price * np.cumprod(1 + np.random.normal(loc=0, scale=0.01, size=n_days))
        for i in range(1, n_days):
            if dates[i].month in [11, 12, 1]:
                price[i] *= 1 + np.random.uniform(0.1, 0.2)
            elif dates[i].month in [5, 6]:
                price[i] *= 1 - np.random.uniform(0.15, 0.2)
        return price
    
    stocks = {}
    stock_names = ['Bullish_1', 'Bullish_2', 'Bearish', 'Sideways_LowVol', 'Sideways_HighVol', 'Cyclical']
    
    generators = [
        (generate_bullish_stock, 60, 0.2),  # Bullish stock 1: 9-13% annual growth
        (generate_bullish_stock, 80, 0.23),  # Bullish stock 2: 15-20% annual growth
        (generate_bearish_stock, 420, -0.25),  # Bearish stock: 10% annual decline
        (generate_sideways_stock, 70, False),  # Sideways stock low volatility
        (generate_sideways_stock, 90, True),  # Sideways stock high volatility
        (generate_cyclical_stock, 150, None)  # Cyclical stock
    ]
    
    for name, (generator, base_price, rate) in zip(stock_names, generators):
        if rate is not None:
            prices = generator(n_days, base_price, rate)
        else:
            prices = generator(n_days, base_price)
        df = pd.DataFrame({'Date': pd.date_range(start='2010-01-01', periods=n_days, freq='B'), 'price': prices})
        df.set_index('Date', inplace=True)
        stocks[name] = df

    return stocks

def generate_data_mix_new():
    stock_data = generate_artificial_stock_data()
    train, test = train_test_split(stock_data, 0.8)
    return train, test

def download_stock_data(tickers=['AAPL', 'MSFT', 'JNJ', 'PG', 'TSLA', 'NFLX', 'KO', 'V'], start_date='2011-01-01', end_date='2024-01-01'):
    stock_data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        stock_data[ticker] = df
    return stock_data

# Format the data to match the artificial data format
def format_stock_data(stock_data):
    formatted_data = {}
    for ticker, df in stock_data.items():
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        formatted_data[ticker] = df[['Close']]
    return formatted_data

def generate_data_from_yfinance():
    tickers = ['AAPL',
        'MSFT',
        'JNJ',
        'PG',
        'TSLA',
        'NFLX',
        'KO',
        'V',
        'GOOGL',
        'AMZN',
        'META',
        'JPM',
        'UNH',
        'HD',
        'VZ',
        'DIS',
        'NVDA',
        'MA',
        'ADBE',
        'IBM']
    stock_data = download_stock_data(tickers, start_date='2012-06-01', end_date='2024-01-01')
    #stock_data = download_stock_data()
    formatted_data = format_stock_data(stock_data)
    train, test, validation  = train_test_split(formatted_data, 0.7)
    return train, test, validation

def generate_crypto():
    tickers = ["BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "XMR-USD", 
           "DASH-USD", "ETC-USD", "ZEC-USD", "DCR-USD", "WAVES-USD"]
    stock_data = download_stock_data(tickers, start_date='2018-01-01', end_date='2024-01-01')
    formatted_data = format_stock_data(stock_data)
    train, test  = train_test_split2(formatted_data, 0.7)
    return train, test

def get_assets_name_list(scenario):
    if scenario == '3bears_1bull':
        return ['Bear_stock_1', 'Bear_stock_2', 'Bear_stock_3', 'Bull_stock_4']
    elif scenario == "mix":
        return ['bull_h_vol', 'bull_l_vol', 'bull_vl_vol', 'bear', 'side', 'side_h_vol']
    elif scenario == "mix_new":
        return ['Bullish_1', 'Bullish_2', 'Bearish', 'Sideways_LowVol', 'Sideways_HighVol', 'Cyclical']
    elif scenario =='1bear_1bull':
        return ['stock1_bull', 'stock2_bear']
    elif scenario =='2bear_1bull':
        return ['stock1_bull', 'stock2_bear', 'stock3_bear']
    elif scenario == 'single_w':
        return ['w_stock']
    elif scenario == "sp":
        return ['AAPL', 'MSFT', 'JNJ', 'PG', 'TSLA', 'NFLX', 'KO', 'V']
    elif scenario == "sp_extended":
        return ['AAPL',
                'MSFT',
                'JNJ',
                'PG',
                'TSLA',
                'NFLX',
                'KO',
                'V',
                'GOOGL',
                'AMZN',
                'META',
                'JPM',
                'UNH',
                'HD',
                'VZ',
                'DIS',
                'NVDA',
                'MA',
                'ADBE',
                'IBM']
    elif scenario == "crypto":
        return ["BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "XMR-USD", 
           "DASH-USD", "ETC-USD", "ZEC-USD", "DCR-USD", "WAVES-USD"]
    else:
        return ['W_stock', 'S_stock', 'Bull_stock', 'Bear_stock']