import numpy as np

from .trading_env import TradingEnv, Actions, Positions
from sklearn import preprocessing, model_selection, feature_selection, ensemble, linear_model, metrics, decomposition
import pandas as pd

class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, features=None, Normalize=True):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.features = features
        self.Normalize = Normalize
        super().__init__(df, window_size)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit


    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        if self.features is not None:
            #print("Features were provided as below:")
            #print(self.features)
            signal_features = self.df[self.features].to_numpy() [self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        else:
            print("No feature was provided")
            diff = np.insert(np.diff(prices/100), 0, 0)
            # Add Ta features here
            signal_features = np.column_stack((prices, diff))

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)

        #normalizing it
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        if self.Normalize:
            signal_features = min_max_scaler.fit_transform(signal_features)
            signal_features[~np.isfinite(signal_features)] = -1


        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += (price_diff)

            # Add short to the reward . If short buy (current_price) should be lower than sell(last_trade_price)
            if self._position == Positions.Short:
                step_reward = step_reward + (-1 * price_diff)

        return step_reward


    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

# how much share you buy times how much you sell them is your money
            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

# reflecting the effect of the trade on the share size basically
            if self._position == Positions.Short:
                shares_sold = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = ((last_trade_price/current_price) * shares_sold * (1 - self.trade_fee_bid_percent)) * last_trade_price

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
