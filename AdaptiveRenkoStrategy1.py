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


class AdvancedSupertrendStrategy1(IStrategy):
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
    buy_ema = IntParameter(30, 180, default=90, space='buy', optimize=True)

    sell_m1 = IntParameter(1, 7, default=4, space='sell', optimize=True)
    sell_m2 = IntParameter(1, 7, default=4, space='sell', optimize=True)
    sell_m3 = IntParameter(1, 7, default=4, space='sell', optimize=True)
    sell_p1 = IntParameter(7, 21, default=14, space='sell', optimize=True)
    sell_p2 = IntParameter(7, 21, default=14, space='sell', optimize=True)
    sell_p3 = IntParameter(7, 21, default=14, space='sell', optimize=True)
    sell_ema = IntParameter(30, 180, default=90, space='sell', optimize=True)

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

    def calculate_supertrend(self, dataframe: DataFrame, period: int, multiplier: float) -> pd.Series:
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
        Adds several different TA indicators to the given DataFrame.
        Calculates all default indicators to ensure correct column names in dataframe.
        """
        # Get default values
        buy_ema_val = 90  # default value from initialization
        sell_ema_val = 90  # default value from initialization
        buy_m1_val = 3    # default values from initialization
        buy_m2_val = 4
        buy_m3_val = 8
        buy_p1_val = 20
        buy_p2_val = 20
        buy_p3_val = 40
        sell_m1_val = 4
        sell_m2_val = 4
        sell_m3_val = 4
        sell_p1_val = 14
        sell_p2_val = 14
        sell_p3_val = 14

        # Stochastic RSI
        dataframe['stoch_rsi'] = ta.momentum.stochrsi(dataframe['close'])

        # EMAs with default values
        dataframe[f'ema_{buy_ema_val}'] = ta.trend.ema_indicator(dataframe['close'], buy_ema_val)
        dataframe[f'ema_{sell_ema_val}'] = ta.trend.ema_indicator(dataframe['close'], sell_ema_val)

        # Buy Supertrends with default values
        supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                  length=buy_p1_val, multiplier=buy_m1_val)
        dataframe[f'supertrend_1_{buy_m1_val}_{buy_p1_val}'] = supertrend[f'SUPERTd_{buy_p1_val}_{float(buy_m1_val):.1f}']

        supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                  length=buy_p2_val, multiplier=buy_m2_val)
        dataframe[f'supertrend_2_{buy_m2_val}_{buy_p2_val}'] = supertrend[f'SUPERTd_{buy_p2_val}_{float(buy_m2_val):.1f}']

        supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                  length=buy_p3_val, multiplier=buy_m3_val)
        dataframe[f'supertrend_3_{buy_m3_val}_{buy_p3_val}'] = supertrend[f'SUPERTd_{buy_p3_val}_{float(buy_m3_val):.1f}']

        # Sell Supertrends with default values
        supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                  length=sell_p1_val, multiplier=sell_m1_val)
        dataframe[f'supertrend_1_{sell_m1_val}_{sell_p1_val}'] = supertrend[f'SUPERTd_{sell_p1_val}_{float(sell_m1_val):.1f}']

        supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                  length=sell_p2_val, multiplier=sell_m2_val)
        dataframe[f'supertrend_2_{sell_m2_val}_{sell_p2_val}'] = supertrend[f'SUPERTd_{sell_p2_val}_{float(sell_m2_val):.1f}']

        supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                  length=sell_p3_val, multiplier=sell_m3_val)
        dataframe[f'supertrend_3_{sell_m3_val}_{sell_p3_val}'] = supertrend[f'SUPERTd_{sell_p3_val}_{float(sell_m3_val):.1f}']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        """
        # Calculate EMA if value is different from default
        if self.buy_ema.value != 90:  # Check against default value
            dataframe[f'ema_{self.buy_ema.value}'] = ta.trend.ema_indicator(dataframe['close'], self.buy_ema.value)

        # Calculate Supertrends if values are different from defaults
        if (self.buy_m1.value != 3) or (self.buy_p1.value != 20):  # Check against default values
            supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                      length=self.buy_p1.value, multiplier=self.buy_m1.value)
            dataframe[f'supertrend_1_{self.buy_m1.value}_{self.buy_p1.value}'] = supertrend[f'SUPERTd_{self.buy_p1.value}_{float(self.buy_m1.value):.1f}']

        if (self.buy_m2.value != 4) or (self.buy_p2.value != 20):  # Check against default values
            supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                      length=self.buy_p2.value, multiplier=self.buy_m2.value)
            dataframe[f'supertrend_2_{self.buy_m2.value}_{self.buy_p2.value}'] = supertrend[f'SUPERTd_{self.buy_p2.value}_{float(self.buy_m2.value):.1f}']

        if (self.buy_m3.value != 8) or (self.buy_p3.value != 40):  # Check against default values
            supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                      length=self.buy_p3.value, multiplier=self.buy_m3.value)
            dataframe[f'supertrend_3_{self.buy_m3.value}_{self.buy_p3.value}'] = supertrend[f'SUPERTd_{self.buy_p3.value}_{float(self.buy_m3.value):.1f}']

        conditions = []
        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['close'] > dataframe[f'ema_{self.buy_ema.value}'])
        conditions.append(dataframe['stoch_rsi'] < self.buy_stoch_rsi.value)

        # Check supertrend signals
        conditions.append(dataframe[f'supertrend_1_{self.buy_m1.value}_{self.buy_p1.value}'] == 1)
        conditions.append(dataframe[f'supertrend_2_{self.buy_m2.value}_{self.buy_p2.value}'] == 1)
        conditions.append(dataframe[f'supertrend_3_{self.buy_m3.value}_{self.buy_p3.value}'] == 1)

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        """
        # Calculate EMA if value is different from default
        if self.sell_ema.value != 90:  # Check against default value
            dataframe[f'ema_{self.sell_ema.value}'] = ta.trend.ema_indicator(dataframe['close'], self.sell_ema.value)

        # Calculate Supertrends if values are different from defaults
        if (self.sell_m1.value != 4) or (self.sell_p1.value != 14):  # Check against default values
            supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                      length=self.sell_p1.value, multiplier=self.sell_m1.value)
            dataframe[f'supertrend_1_{self.sell_m1.value}_{self.sell_p1.value}'] = supertrend[f'SUPERTd_{self.sell_p1.value}_{float(self.sell_m1.value):.1f}']

        if (self.sell_m2.value != 4) or (self.sell_p2.value != 14):  # Check against default values
            supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                      length=self.sell_p2.value, multiplier=self.sell_m2.value)
            dataframe[f'supertrend_2_{self.sell_m2.value}_{self.sell_p2.value}'] = supertrend[f'SUPERTd_{self.sell_p2.value}_{float(self.sell_m2.value):.1f}']

        if (self.sell_m3.value != 4) or (self.sell_p3.value != 14):  # Check against default values
            supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                      length=self.sell_p3.value, multiplier=self.sell_m3.value)
            dataframe[f'supertrend_3_{self.sell_m3.value}_{self.sell_p3.value}'] = supertrend[f'SUPERTd_{self.sell_p3.value}_{float(self.sell_m3.value):.1f}']

        conditions = []
        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['close'] < dataframe[f'ema_{self.sell_ema.value}'])
        conditions.append(dataframe['stoch_rsi'] > self.sell_stoch_rsi.value)

        # Check supertrend signals
        conditions.append(dataframe[f'supertrend_1_{self.sell_m1.value}_{self.sell_p1.value}'] == -1)
        conditions.append(dataframe[f'supertrend_2_{self.sell_m2.value}_{self.sell_p2.value}'] == -1)
        conditions.append(dataframe[f'supertrend_3_{self.sell_m3.value}_{self.sell_p3.value}'] == -1)

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'sell'] = 1

        return dataframe
