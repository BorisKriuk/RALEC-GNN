import numpy as np
import pandas as pd


class PerformanceMetricsCalculator:
    def __init__(self, risk_free_rate=0.0):
        self.equity_curve_data = []
        self.risk_free_rate = risk_free_rate
        self.metrics = {}

    def add_equity_point(self, timestamp, equity_value):
        self.equity_curve_data.append({'timestamp': timestamp, 'equity': equity_value})

    def _get_equity_dataframe(self):
        if not self.equity_curve_data:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_curve_data)
        df = df.set_index('timestamp')
        df['equity'] = pd.to_numeric(df['equity'])
        return df

    def _get_annualization_factor(self, interval):
        if interval == '1m':
            return 252 * 24 * 60
        elif interval == '1h':
            return 252 * 24
        elif interval == '1d':
            return 252
        elif interval.endswith('min'):
            minutes = int(interval[:-3])
            return (252 * 24 * 60) / minutes
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            return (252 * 24) / hours
        elif interval.endswith('d'):
            days = int(interval[:-1])
            return 252 / days
        else:
            return 252

    def calculate_maximum_drawdown(self):
        df = self._get_equity_dataframe()
        if df.empty:
            return 0.0
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['peak'] - df['equity']) / df['peak']
        max_drawdown = df['drawdown'].max() * 100
        self.metrics['maximum_drawdown'] = max_drawdown
        return max_drawdown

    def calculate_total_return(self):
        if not self.equity_curve_data:
            return 0.0
        initial_equity = self.equity_curve_data[0]['equity']
        final_equity = self.equity_curve_data[-1]['equity']
        if initial_equity == 0:
            return 0.0
        total_return = ((final_equity - initial_equity) / initial_equity) * 100
        self.metrics['total_return'] = total_return
        return total_return

    def calculate_annualized_return(self, interval):
        if not self.equity_curve_data or len(self.equity_curve_data) < 2:
            return 0.0
        df = self._get_equity_dataframe()
        if df.empty:
            return 0.0
        first_date = df.index.min()
        last_date = df.index.max()
        total_duration = last_date - first_date

        if total_duration.total_seconds() <= 0:
            return 0.0

        num_years = total_duration.total_seconds() / (365.25 * 24 * 3600)

        total_return_decimal = self.calculate_total_return() / 100

        if num_years == 0:
            return 0.0

        annualized_return = ((1 + total_return_decimal) ** (1 / num_years) - 1) * 100

        self.metrics['annualized_return'] = annualized_return
        return annualized_return

    def calculate_volatility(self, interval):
        df = self._get_equity_dataframe()
        if df.empty or len(df) < 2:
            return 0.0
        df['returns'] = df['equity'].pct_change().fillna(0)
        periods_per_year = self._get_annualization_factor(interval)
        annualized_volatility = df['returns'].std() * np.sqrt(periods_per_year)
        self.metrics['volatility'] = annualized_volatility
        return annualized_volatility

    def calculate_sharpe_ratio(self, interval):
        if not self.equity_curve_data or len(self.equity_curve_data) < 2:
            return 0.0
        annualized_return = (self.calculate_annualized_return(interval) / 100)
        excess_return = annualized_return - self.risk_free_rate
        annualized_volatility = self.calculate_volatility(interval)
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility != 0 else 0.0
        self.metrics['sharpe_ratio'] = sharpe_ratio
        return sharpe_ratio

    def calculate_sortino_ratio(self, interval):
        df = self._get_equity_dataframe()
        if df.empty or len(df) < 2:
            return 0.0
        df['returns'] = df['equity'].pct_change().fillna(0)
        annualized_return = (self.calculate_annualized_return(interval) / 100)
        excess_return = annualized_return - self.risk_free_rate

        periods_per_year = self._get_annualization_factor(interval)

        downside_returns = df[df['returns'] < self.risk_free_rate]['returns']
        if downside_returns.empty:
            return 0.0

        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)

        sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0.0

        self.metrics['sortino_ratio'] = sortino_ratio
        return sortino_ratio

    def calculate_value_at_risk(self, confidence_level=0.95):
        df = self._get_equity_dataframe()
        if df.empty or len(df) < 2:
            return 0.0
        df['returns'] = df['equity'].pct_change().fillna(0)
        var = -np.percentile(df['returns'], 100 * (1 - confidence_level))
        self.metrics['value_at_risk'] = var
        return var