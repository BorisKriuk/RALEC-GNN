class PortfolioManager:
    def __init__(self, initial_cash=100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.transactions = []
        self.equity_curve = []

    def update_portfolio(self, order, executed_price, transaction_cost, current_timestamp):
        if order.symbol not in self.positions:
            self.positions[order.symbol] = {'quantity': 0.0, 'avg_price': 0.0}

        cost = executed_price * order.quantity

        if order.side == 'buy':
            self.cash -= (cost + transaction_cost)
            current_qty = self.positions[order.symbol]['quantity']
            current_value = current_qty * self.positions[order.symbol]['avg_price']
            new_qty = current_qty + order.quantity
            new_avg_price = (current_value + cost) / new_qty if new_qty > 0 else 0.0
            self.positions[order.symbol]['quantity'] = new_qty
            self.positions[order.symbol]['avg_price'] = new_avg_price

        elif order.side == 'sell':
            self.cash += (cost - transaction_cost)
            self.positions[order.symbol]['quantity'] -= order.quantity
            if self.positions[order.symbol]['quantity'] == 0:
                self.positions[order.symbol]['avg_price'] = 0.0

        self.transactions.append({
            'order_id': order.order_id,
            'timestamp': current_timestamp,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'executed_price': executed_price,
            'transaction_cost': transaction_cost,
            'cash_balance': self.cash,
            'position_qty': self.positions[order.symbol]['quantity'] if order.symbol in self.positions else 0
        })

    def calculate_equity(self, current_market_prices, current_timestamp):
        total_assets_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_market_prices and position['quantity'] > 0:
                total_assets_value += position['quantity'] * current_market_prices[symbol]

        current_equity = self.cash + total_assets_value
        self.equity_curve.append({'timestamp': current_timestamp, 'equity': current_equity})
        return current_equity

    def calculate_pnl(self, current_market_prices):
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            if position['quantity'] > 0 and symbol in current_market_prices:
                unrealized_pnl += (current_market_prices[symbol] - position['avg_price']) * position['quantity']

        if self.equity_curve:
            latest_equity_point = self.equity_curve[-1]
            total_pnl = latest_equity_point['equity'] - self.initial_cash
        else:
            total_pnl = self.cash - self.initial_cash

        return {
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl
        }