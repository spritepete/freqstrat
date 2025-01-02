# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                              IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import ta
import pandas_ta as pda


class AdvancedSupertrendStrategy(IStrategy):
    """
    Advanced Supertrend Strategy with hyperoptable parameters for multiple timeframes
    and multipliers. This allows finding optimal parameters through hyperopt.
    """
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 100  # inactive
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.99  # inactive

    # Trailing stoploss
    trailing_stop = False

    # Hyperoptable parameters
    buy_m1 = IntParameter(1, 8, default=3, space='buy', optimize=True)
    buy_m2 = IntParameter(1, 10, default=4, space='buy', optimize=True)
    buy_m3 = IntParameter(1, 14, default=8, space='buy', optimize=True)
    buy_p1 = IntParameter(7, 51, default=20, space='buy', optimize=True)
    buy_p2 = IntParameter(7, 101, default=20, space='buy', optimize=True)
    buy_p3 = IntParameter(7, 151, default=40, space='buy', optimize=True)
    
    sell_m1 = IntParameter(1, 7, default=4, space='sell', optimize=True)
    sell_m2 = IntParameter(1, 7, default=4, space='sell', optimize=True)
    sell_m3 = IntParameter(1, 7, default=4, space='sell', optimize=True)
    sell_p1 = IntParameter(7, 21, default=14, space='sell', optimize=True)
    sell_p2 = IntParameter(7, 21, default=14, space='sell', optimize=True)
    sell_p3 = IntParameter(7, 21, default=14, space='sell', optimize=True)

    # Additional parameters
    buy_stoch_rsi = DecimalParameter(0.5, 1, decimals=3, default=0.8, space="buy")
    sell_stoch_rsi = DecimalParameter(0, 0.5, decimals=3, default=0.2, space="sell")

    # Strategy parameters
    timeframe = '1h'
    process_only_new_candles = False
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    startup_candle_count: int = 150  # Increased due to longer lookback periods

    # Order settings
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    def calculate_supertrend(self, dataframe: DataFrame, period: int, multiplier: float, prefix: str) -> pd.Series:
        """
        Calculate Supertrend indicator with proper error handling
        """
        try:
            supertrend = pda.supertrend(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                length=period,
                multiplier=multiplier
            )
            direction_col = f'SUPERTd_{period}_{multiplier:.1f}'
            if direction_col in supertrend:
                return supertrend[direction_col]
            # Fallback if column name doesn't match
            direction_cols = [col for col in supertrend.columns if col.startswith('SUPERTd_')]
            if direction_cols:
                return supertrend[direction_cols[0]]
            return pd.Series(0, index=dataframe.index)  # Default fallback
        except Exception as e:
            print(f"Error calculating Supertrend for period={period}, multiplier={multiplier}: {str(e)}")
            return pd.Series(0, index=dataframe.index)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        # Stochastic RSI
        dataframe['stoch_rsi'] = ta.momentum.stochrsi(dataframe['close'])
        
        # EMA
        dataframe['ema90'] = ta.trend.ema_indicator(dataframe['close'], 90)

        # Calculate Supertrend for buy signals
        for multiplier in self.buy_m1.range:
            for period in self.buy_p1.range:
                col_name = f'supertrend_1_buy_{multiplier}_{period}'
                dataframe[col_name] = self.calculate_supertrend(
                    dataframe, period, float(multiplier), '1_buy'
                )

        for multiplier in self.buy_m2.range:
            for period in self.buy_p2.range:
                col_name = f'supertrend_2_buy_{multiplier}_{period}'
                dataframe[col_name] = self.calculate_supertrend(
                    dataframe, period, float(multiplier), '2_buy'
                )

        for multiplier in self.buy_m3.range:
            for period in self.buy_p3.range:
                col_name = f'supertrend_3_buy_{multiplier}_{period}'
                dataframe[col_name] = self.calculate_supertrend(
                    dataframe, period, float(multiplier), '3_buy'
                )

        # Calculate Supertrend for sell signals
        for multiplier in self.sell_m1.range:
            for period in self.sell_p1.range:
                col_name = f'supertrend_1_sell_{multiplier}_{period}'
                dataframe[col_name] = self.calculate_supertrend(
                    dataframe, period, float(multiplier), '1_sell'
                )

        for multiplier in self.sell_m2.range:
            for period in self.sell_p2.range:
                col_name = f'supertrend_2_sell_{multiplier}_{period}'
                dataframe[col_name] = self.calculate_supertrend(
                    dataframe, period, float(multiplier), '2_sell'
                )

        for multiplier in self.sell_m3.range:
            for period in self.sell_p3.range:
                col_name = f'supertrend_3_sell_{multiplier}_{period}'
                dataframe[col_name] = self.calculate_supertrend(
                    dataframe, period, float(multiplier), '3_sell'
                )

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        """
        conditions = []
        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['close'] > dataframe['ema90'])
        conditions.append(dataframe['stoch_rsi'] < self.buy_stoch_rsi.value)

        # Check supertrend signals for current parameter values
        conditions.append(
            dataframe[f'supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}'] == 1
        )
        conditions.append(
            dataframe[f'supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}'] == 1
        )
        conditions.append(
            dataframe[f'supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}'] == 1
        )

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        """
        conditions = []
        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['stoch_rsi'] > self.sell_stoch_rsi.value)

        # Check supertrend signals for current parameter values
        conditions.append(
            dataframe[f'supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}'] == -1
        )
        conditions.append(
            dataframe[f'supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}'] == -1
        )
        conditions.append(
            dataframe[f'supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}'] == -1
        )

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'sell'] = 1

        return dataframe