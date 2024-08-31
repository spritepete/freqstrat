# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from stocktrends import indicators
from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicators import ichimoku

#renkoBegin = 4
#renkoEnd = 62
procent_buy = 1
procent_sell = 30

class Awesome_ichimoku_renk_25_backtest(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    '''minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }'''

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.135

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal ticker interval for the strategy.
    ticker_interval = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        #renko
        #dataframe['date'] = pd.to_datetime(dataframe['timestamp'], unit='s')

        def renkoDataFrame(ohlc_data, brickSize):
            df = ohlc_data.copy()    
            renko = indicators.Renko(df)
            #renko.brick_size = optimal_brick
            renko.brick_size = brickSize
            renko.chart_type = indicators.Renko.PERIOD_CLOSE
            data = renko.get_ohlc_data()
            data['bar_num'] = np.where(data['uptrend'] == True, 1, np.where(data['uptrend'] == False, -1, 0))
            for i in range(1,len(data["bar_num"])):
                if data["bar_num"][i]>0 and data["bar_num"][i-1]>0:
                    #data["bar_num"][i]+=data["bar_num"][i-1]
                    data.loc[i,'bar_num'] += data.loc[i-1,'bar_num']
                elif data["bar_num"][i]<0 and data["bar_num"][i-1]<0:
                    data.loc[i,'bar_num'] += data.loc[i-1,'bar_num']
            data.drop_duplicates(subset = 'date', keep = 'last', inplace = True)
            df['bar_num'] = df['date'].map(data.set_index('date')['bar_num'].to_dict())
            df['bar_num'].fillna(method='ffill',inplace=True)
            df['renko_close'] = df['date'].map(data.set_index('date')['close'].to_dict())
            df['renko_close'].fillna(method='ffill',inplace=True)
            return df#['bar_num']

        brickSizeInitial = dataframe['close'].iloc[-1]
        #brickSizeInitial = 590.5
        #for i in range(renkoBegin,renkoEnd):
        #popravil initial blricksize
        dataframe[[f'bar_num{procent_buy}',f'renko_close{procent_buy}']] = renkoDataFrame(dataframe, brickSize = (brickSizeInitial/250)*procent_buy)[['bar_num','renko_close']]
        #dataframe[[f'bar_num{procent_buy}',f'renko_close{procent_buy}']] = renkoDataFrame(dataframe, brickSize = brickSizeInitial*procent_buy)[['bar_num','renko_close']]
        
        #dataframe[f'renko_close{procent_buy}'] = renkoDataFrame(dataframe, brickSize = (brickSizeInitial/1000)*procent_buy)['renko_close']
        
        dataframe[[f'bar_num{procent_sell}',f'renko_close{procent_sell}']] = renkoDataFrame(dataframe, brickSize = (brickSizeInitial/250)*procent_sell)[['bar_num','renko_close']]
        #dataframe[[f'bar_num{procent_sell}',f'renko_close{procent_sell}']] = renkoDataFrame(dataframe, brickSize = brickSizeInitial*procent_buy)[['bar_num','renko_close']]
        #dataframe[f'bar_num{procent_sell}'] = renkoDataFrame(dataframe, brickSize = (brickSizeInitial/1000)*procent_sell)['bar_num']

        def ichiDataFrame(ohlc_data, param):     
            ichi = ichimoku(ohlc_data, conversion_line_period=4+(param), base_line_periods=13+(3*param),laggin_span=26+(6*param), displacement=13+(3*param))
            data = pd.DataFrame.from_dict(ichi)
            #data = ichi['kijun_sen']
            conditions = [
                (data['kijun_sen'] > data['kijun_sen'].shift(1)),
                (data['kijun_sen'] < data['kijun_sen'].shift(1)),
                (data['kijun_sen'] == data['kijun_sen'].shift(1))]
            choices = [1, -1, 0]
            data['kijun_trend'] = np.select(conditions, choices)
            data['kijun_trend'] = data['kijun_trend'].replace(to_replace=0, method='ffill')
            return data#['trend']
            
            

        for paramterIchi in range(1,300):
            #dataframe[[f'kijun_trend{paramterIchi}',f'tenkan_sen1{paramterIchi}']] = ichiDataFrame(dataframe, paramterIchi)[['kijun_trend','tenkan_sen']]
            #dataframe[f'kijun_trend{paramterIchi}'] = ichiDataFrame(dataframe, paramterIchi)['kijun_trend']
            dataframe[[f'kijun_trend{paramterIchi}',f'tenkan_sen{paramterIchi}',f'kijun_sen{paramterIchi}',f'cloud_green{paramterIchi}',f'senkou_span_a{paramterIchi}',f'senkou_span_b{paramterIchi}']] = ichiDataFrame(dataframe, paramterIchi)[['kijun_trend','tenkan_sen','kijun_sen','cloud_green','senkou_span_a','senkou_span_b']]
            

        #V dataframe dodan renko_open in renko_close
        #dataframe['renko_close'] = dataframe['date'].map(data.set_index('date')['close'].to_dict())
        #dataframe['renko_close'].fillna(method='ffill',inplace=True)
        #dataframe['renko_open'] = dataframe['date'].map(data.set_index('date')['open'].to_dict())
        #dataframe['renko_open'].fillna(method='ffill',inplace=True)
        



        #btc = dataframe.copy()
        #btc['date'] = btc['date'].dt.tz_convert(None)
        #ohlc_renko = btc.merge(data.loc[:,["date", "bar_num"]],how="outer",on="date")
        #dataframe['bar_num'] = dataframe['date'].map(data['bar_num'])

        #ohlc_renko["bar_num"].fillna(method='ffill',inplace=True)
        #dataframe['bar_num'] = ohlc_renko["bar_num"]

        #heikinashi = qtpylib.heikinashi(dataframe)
        #dataframe['ha_open'] = heikinashi['open']
        #dataframe['ha_close'] = heikinashi['close']

        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe[f'bar_num{procent_buy}'] >= 1) & # green bar
                (dataframe['close'] > dataframe['senkou_span_a70']) &
                (dataframe['close'] > dataframe['senkou_span_b70'])&
                (dataframe['kijun_trend70'] >= 1)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe[f'bar_num{procent_sell}'] <= -1) & # green bar
                (dataframe['kijun_trend127'] <= -1)&
		        (dataframe['close'] < dataframe['senkou_span_a127'])&
		        (dataframe['close'] < dataframe['senkou_span_b127'])
            ),
            'sell'] = 1
        return dataframe
