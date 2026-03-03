import os

import numpy as np
import pandas as pd
import requests


class EODDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_historical_klines(self, symbol='BTC-USD.CC', interval='d', start_str=None, end_str=None):
        base_url = "https://eodhistoricaldata.com/api/"
        endpoint = f"eod/{symbol}"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "period": interval
        }

        if start_str:
            params["from"] = start_str
        if end_str:
            params["to"] = end_str

        request_url = f"{base_url}{endpoint}"
        
        try:
            response = requests.get(request_url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            df = df.rename(columns={'date': 'Open time'})
            df['Open time'] = pd.to_datetime(df['Open time'])
            df = df.set_index('Open time')

            standard_ohlcv_names = ['Open', 'High', 'Low', 'Close', 'Volume']
            eod_api_names = ['open', 'high', 'low', 'close', 'volume']

            for std_name, api_name in zip(standard_ohlcv_names, eod_api_names):
                if api_name in df.columns:
                    df = df.rename(columns={api_name: std_name})
                else:
                    df[std_name] = np.nan

            for col in standard_ohlcv_names:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df[standard_ohlcv_names]
            return df

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - Response: {response.text}")
            return pd.DataFrame()
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected error occurred during API request: {req_err}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred during data processing: {e}")
            return pd.DataFrame()

    def save_data(self, dataframe, symbol, interval, path='data'):
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{path}/{symbol}_{interval}.csv"
        dataframe.to_csv(filename)