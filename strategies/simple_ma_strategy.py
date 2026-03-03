import pandas as pd

from backtesting_engine.backtester.strategy_interface import StrategyInterface


class SimpleMovingAverageStrategy(StrategyInterface.TradingStrategy):
    def __init__(self, short_window=10, long_window=20, quantity_per_trade=0.005):
        self.short_window = short_window
        self.long_window = long_window
        self.quantity_per_trade = quantity_per_trade
        self.df_history = pd.DataFrame()
        self.position = 0

    def generate_signal(self, ohlcv_data: pd.DataFrame) -> dict:
        if self.df_history.empty:
            self.df_history = ohlcv_data.copy()
        else:
            self.df_history = pd.concat([self.df_history, ohlcv_data])

        if len(self.df_history) < self.long_window:
            return {'signal': 'hold', 'amount': 0}

        close_col = 'Close_close' if 'Close_close' in self.df_history.columns else 'Close'
        self.df_history['SMA_Short'] = self.df_history[close_col].rolling(window=self.short_window).mean()
        self.df_history['SMA_Long'] = self.df_history[close_col].rolling(window=self.long_window).mean()

        latest_sma_short = self.df_history['SMA_Short'].iloc[-1]
        latest_sma_long = self.df_history['SMA_Long'].iloc[-1]

        signal = 'hold'
        amount = 0

        if latest_sma_short > latest_sma_long and self.position == 0:
            signal = 'buy'
            amount = self.quantity_per_trade
            self.position = 1
        elif latest_sma_short < latest_sma_long and self.position == 1:
            signal = 'sell'
            amount = self.quantity_per_trade
            self.position = 0

        return {'signal': signal, 'amount': amount}