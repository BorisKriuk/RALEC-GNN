import pandas as pd


class DataPreprocessor:
    def __init__(self):
        pass

    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            if 'Open time' in df.columns:
                df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
                df = df.set_index('Open time')
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            else:
                raise ValueError("Date column ('Open time' or 'Date') not found.")

            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                            'Quote asset volume', 'Number of trades',
                            'Taker buy base asset volume', 'Taker buy quote asset volume']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return pd.DataFrame()

    def clean_data(self, dataframe):
        dataframe.ffill(inplace=True)
        critical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        dataframe.dropna(subset=critical_cols, inplace=True)
        return dataframe

    def resample_data(self, dataframe, interval):
        close_col = 'Close_close' if 'Close_close' in dataframe.columns else 'Close'

        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        if isinstance(dataframe.columns, pd.MultiIndex):
            dataframe.columns = ['_'.join(col).strip() for col in dataframe.columns.values]

        resampled_ohlc = dataframe[ohlc_cols].resample(interval).ohlc()
        resampled_ohlc.columns = ['_'.join(col).strip() for col in resampled_ohlc.columns.values]

        other_cols = ['Volume', 'Quote asset volume', 'Number of trades',
                      'Taker buy base asset volume', 'Taker buy quote asset volume']
        existing_other_cols = [col for col in other_cols if col in dataframe.columns]
        resampled_others = dataframe[existing_other_cols].resample(interval).sum()

        resampled_df = pd.concat([resampled_ohlc, resampled_others], axis=1)
        resampled_df.dropna(inplace=True)
        return resampled_df