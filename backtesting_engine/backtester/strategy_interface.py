from abc import ABC

import pandas as pd
from sklearn.naive_bayes import abstractmethod


class StrategyInterface:
    class TradingStrategy(ABC):
        """Abstract base class for all trading strategies."""

        @abstractmethod
        def generate_signal(self, ohlcv_data: pd.DataFrame) -> dict:
            """
            Generates trading signals based on the provided OHLCV data.

            Args:
                ohlcv_data (pd.DataFrame): DataFrame containing OHLCV data.

            Returns:
                dict: A dictionary containing trading signals. Expected format:
                      {'signal': 'buy'/'sell'/'hold', 'amount': quantity/percentage}
            """
            raise NotImplementedError("Subclasses must implement the generate_signal method")
