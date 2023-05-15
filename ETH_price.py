import sys
import numpy as np
import requests
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

""" Part of getting historical data """


def clean_price(data):
    """ Function to handle price change """

    for price_elem in range(len(data)):
        if '.' in data[price_elem]:
            data[price_elem] = (data[price_elem][0:data[price_elem].index('.')])
        data[price_elem] = data[price_elem].replace(',', '.')
    return data.astype(float)


def clean_percent(data):
    """ Function to handle percentage change """

    for percent_elem in range(len(data)):
        if '%' in data[percent_elem]:
            data[percent_elem] = data[percent_elem][0:data[percent_elem].index('%')]
    return data.astype(float)


def data_clean():
    """ Function for reading and cleaning data """

    data_btc = pd.read_csv('Bitcoin Historical Data - Investing.com.csv')
    data_eth = pd.read_csv('Ethereum Historical Data - Investing.com.csv')
    data_btc = data_btc[['Date', 'Price', 'Change %']]
    data_eth = data_eth[['Date', 'Price', 'Change %']]
    data_btc['Price'] = clean_price(data_btc['Price'])
    data_eth['Price'] = clean_price(data_eth['Price'])
    data_btc['Change %'] = clean_percent(data_btc['Change %'])
    data_eth['Change %'] = clean_percent(data_eth['Change %'])
    return data_btc, data_eth


""" Part of creating a regression """


def regression(X_train, y_train, X_test, y_test):
    """ Train linear regression """

    model = LinearRegression()
    model.fit(X=X_train, y=y_train)
    """ If we need graphs, we can use mse and mae for that """
    mse = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))
    mae = mean_absolute_error(y_true=y_test, y_pred=model.predict(X_test))
    return model


""" Part of the ETH price estimate from BTC """


def check_price(model):
    """ The function of finding and comparing the price of ETH """

    btc = pd.DataFrame()
    eth = pd.DataFrame()
    current_time = time.perf_counter()
    now_time = time.perf_counter()
    while True:

        btc_key = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        eth_key = "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT"

        data = requests.get(btc_key)
        data = data.json()
        realtime = time.perf_counter() - current_time
        btc[realtime] = [data['price']]

        data = requests.get(eth_key)
        data = data.json()
        eth[realtime] = [data['price']]

        if len(eth.columns) > 1:
            btc_difference = (100 * (float(btc[btc.columns[len(btc.columns) - 1]]) - float(
                btc[btc.columns[len(btc.columns) - 2]])) / float(btc[btc.columns[len(btc.columns) - 2]]))
            eth_difference = (100 * (float(eth[eth.columns[len(eth.columns) - 1]]) - float(
                eth[eth.columns[len(eth.columns) - 2]])) / float(eth[eth.columns[len(eth.columns) - 2]]))
            print(f'ETH current price {float(eth[eth.columns[len(eth.columns) - 1]])}')
            print(f'Price change relative to the previous value: {eth_difference}%')
            print(f'Price change taking into account BTC price: '
                  f'{eth_difference - float(model.predict([[btc_difference]]))}%')
            print(f'-'* 50)

        if time.perf_counter() - now_time > 3600:
            last_value_eth = eth.columns[len(eth.columns) - 1]
            now_time = time.perf_counter()
            for elem_eth in reversed(eth.columns):
                if float(last_value_eth) - float(elem_eth) >= 3599:
                    previous_value_eth = elem_eth
                    break
            change = 100 * (
                        (float(eth[last_value_eth]) - float(eth[previous_value_eth])) / float(eth[previous_value_eth]))
            if change >= 1:
                print(f'-' * 50)
                print(f'Price changed by: {change}% ')
                print(f'-' * 50)



""" Start part """

if __name__ == '__main__':
    data_btc, data_eth = data_clean()
    X = np.array([[elem] for elem in data_btc['Change %']])
    y = np.array(data_eth['Change %'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    model = regression(X_train, y_train, X_test, y_test)
    check_price(model)
