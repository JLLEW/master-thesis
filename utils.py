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
    #print(f"observation: {data}")
    #print(f"type of obs: {type(data)}")
    #print(f"data keys: {data.keys()}")
  
    df = pd.DataFrame(data)
    returns = np.log(df).diff()
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

def generate_test_data(scenario, seed):
    random.seed(seed)
    if scenario == '3bears_1bull':
        assets = generate_3bears_1bull()
    elif scenario == 'mix':
        assets = generate_mix()
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

def generate_train_data(scenario, seed):
    random.seed(seed)
    if scenario == '3bears_1bull':
        assets = generate_3bears_1bull()
    elif scenario == 'mix':
        assets = generate_mix()
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
    bull_high_vol = data_generator(3000, [0.45, 0.55], random.randint(230, 400), -4, 4)
    bull_low_vol = data_generator(3000, [0.3, 0.7], random.randint(120, 200), -1, 1)
    bull_very_low_vol = data_generator(3000, [0.43, 0.57], random.randint(120, 200), -0.1, 0.3)
    bear = data_generator(3000, [0.6, 0.4], random.randint(12000, 17000), -3, 2.5)
    side = data_generator(3000, [0.5, 0.5], random.randint(25, 40), -1, 1)
    side_high_vol = data_generator(3000, [0.5, 0.5], random.randint(15000, 50000), -2, 2)
    
    assets = {
        'bull_h_vol': bull_high_vol,
        'bull_l_vol': bull_low_vol,
        'bull_vl_vol': bull_very_low_vol,
        'bear': bear,
        'side': side,
        'side_h_vol': side_high_vol
    }

    return assets


def get_assets_name_list(scenario):
    if scenario == '3bears_1bull':
        return ['Bear_stock_1', 'Bear_stock_2', 'Bear_stock_3', 'Bull_stock_4']
    elif scenario == "mix":
        return ['bull_h_vol', 'bull_l_vol', 'bull_vl_vol', 'bear', 'side', 'side_h_vol']
    elif scenario =='1bear_1bull':
        return ['stock1_bull', 'stock2_bear']
    elif scenario =='2bear_1bull':
        return ['stock1_bull', 'stock2_bear', 'stock3_bear']
    elif scenario == 'single_w':
        return ['w_stock']
    else:
        return ['W_stock', 'S_stock', 'Bull_stock', 'Bear_stock']