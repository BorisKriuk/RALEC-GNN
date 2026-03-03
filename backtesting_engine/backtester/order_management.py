from datetime import datetime
import uuid


class OrderManagementSystem:
    def __init__(self):
        self.orders = {}

    class Order:
        def __init__(self, symbol, order_type, side, quantity, price=None, status='open'):
            self.order_id = str(uuid.uuid4()) # Unique identifier for the order
            self.symbol = symbol
            self.order_type = order_type # e.g., 'market', 'limit'
            self.side = side             # e.g., 'buy', 'sell'
            self.quantity = quantity
            self.price = price           # For market orders, this can be None initially, filled upon execution
            self.status = 'open'         # e.g., 'open', 'filled', 'cancelled'
            self.timestamp = datetime.datetime.now()
            self.executed_price = None
            self.transaction_cost = None

        def __repr__(self):
            return (f"Order(id={self.order_id[:8]}..., symbol='{self.symbol}', type='{self.order_type}', "
                    f"side='{self.side}', qty={self.quantity}, price={self.price}, status='{self.status}', "
                    f"time={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})")

    def add_order(self, symbol, order_type, side, quantity, price=None):
        order = self.Order(symbol, order_type, side, quantity, price)
        self.orders[order.order_id] = order
        return order

    def cancel_order(self, order_id):
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == 'open':
                order.status = 'cancelled'
                return True
            else:
                return False
        else:
            return False

    def get_order_status(self, order_id):
        if order_id in self.orders:
            return self.orders[order_id].status
        return None

    def get_all_orders(self):
        return self.orders