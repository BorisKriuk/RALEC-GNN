class ExecutionSimulator:
    def __init__(self, transaction_cost_rate=0.001, slippage_factor=0.0002, market_impact_factor=0.0001):
        self.transaction_cost_rate = transaction_cost_rate
        self.slippage_factor = slippage_factor
        self.market_impact_factor = market_impact_factor

    def execute_order(self, order, current_market_price):
        if order.status != 'open':
            return None, None

        executed_price = current_market_price

        if order.side == 'buy':
            slippage_amount = current_market_price * self.slippage_factor
            market_impact_amount = self.market_impact_factor * order.quantity
            executed_price = current_market_price + slippage_amount + market_impact_amount
        elif order.side == 'sell':
            slippage_amount = current_market_price * self.slippage_factor
            market_impact_amount = self.market_impact_factor * order.quantity
            executed_price = current_market_price - slippage_amount - market_impact_amount

        transaction_cost = executed_price * order.quantity * self.transaction_cost_rate

        order.status = 'filled'
        order.executed_price = executed_price
        order.transaction_cost = transaction_cost

        return executed_price, transaction_cost