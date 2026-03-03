class BacktestRunner:
    def __init__(self, data_preprocessor, strategy, oms, execution_simulator, portfolio_manager, performance_calculator):
        self.data_preprocessor = data_preprocessor
        self.strategy = strategy
        self.oms = oms
        self.execution_simulator = execution_simulator
        self.portfolio_manager = portfolio_manager
        self.performance_calculator = performance_calculator

    def run_backtest(self, ohlcv_data_path, symbol, interval='1m', start_date=None, end_date=None):
        raw_data = self.data_preprocessor.load_data(ohlcv_data_path)
        if raw_data.empty:
            print("No data loaded. Exiting backtest.")
            return None, None, None

        cleaned_data = self.data_preprocessor.clean_data(raw_data.copy())
        processed_data = self.data_preprocessor.resample_data(cleaned_data.copy(), interval)
        
        if 'Close_close' not in processed_data.columns and 'Close' in processed_data.columns:
             processed_data = processed_data.rename(columns={'Open': 'Open_open', 'High': 'High_high', 'Low': 'Low_low', 'Close': 'Close_close'})

        if start_date:
            processed_data = processed_data[processed_data.index >= start_date]
        if end_date:
            processed_data = processed_data[processed_data.index <= end_date]

        if processed_data.empty or len(processed_data) < 2:
            print("No sufficient data after date filtering for backtest. Exiting backtest.")
            return None, None, None

        initial_timestamp = processed_data.index[0]
        self.portfolio_manager.equity_curve.append({'timestamp': initial_timestamp, 'equity': self.portfolio_manager.initial_cash})
        self.performance_calculator.add_equity_point(initial_timestamp, self.portfolio_manager.initial_cash)

        for i, (timestamp, row) in enumerate(processed_data.iterrows()):
            current_market_price = row['Close_close']
            strategy_signals = self.strategy.generate_signal(row.to_frame().T)

            signal = strategy_signals.get('signal')
            quantity = strategy_signals.get('amount')

            if signal and signal != 'hold' and quantity > 0:
                order_side = signal
                order = self.oms.add_order(symbol, 'market', order_side, quantity)

                executed_price, transaction_cost = self.execution_simulator.execute_order(order, current_market_price)

                if executed_price is not None:
                    self.portfolio_manager.update_portfolio(order, executed_price, transaction_cost, timestamp)

            current_prices_for_equity = {symbol: current_market_price}
            equity = self.portfolio_manager.calculate_equity(current_prices_for_equity, timestamp)
            self.performance_calculator.add_equity_point(timestamp, equity)

        final_equity = self.portfolio_manager.cash
        for sym, pos in self.portfolio_manager.positions.items():
            if pos['quantity'] > 0:
                final_equity += pos['quantity'] * processed_data.iloc[-1]['Close_close']

        pnl_results = self.portfolio_manager.calculate_pnl(current_prices_for_equity)

        self.performance_calculator.calculate_maximum_drawdown()
        self.performance_calculator.calculate_total_return()
        self.performance_calculator.calculate_annualized_return(interval)
        self.performance_calculator.calculate_volatility(interval)
        self.performance_calculator.calculate_sharpe_ratio(interval)
        self.performance_calculator.calculate_sortino_ratio(interval)
        self.performance_calculator.calculate_value_at_risk()

        return self.portfolio_manager.equity_curve, self.portfolio_manager.transactions, self.performance_calculator.metrics
