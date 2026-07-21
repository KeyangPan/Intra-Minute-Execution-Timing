from itertools import product

import pandas as pd
import numpy as np

FACTOR_HYPERPARAMETERS = {
    'obi_weighted': {'level_weights': (50, 30, 20)},
    'spread': {},
    'spread_deviation': {'look_back_ticks': 100},
    'ask_px_momentum': {'lag_ticks': 100},
    'bid_px_momentum': {'lag_ticks': 100},
    'vwap_mid_deviation_ask': {'look_back_ticks': 100,
                               'level_weights': (85, 10, 5)},
    'vwap_mid_deviation_bid': {'look_back_ticks': 100,
                               'level_weights': (85, 10, 5)},
}

class Factors:
    def __init__(self,
                 data,
                 data_depth = 3,
                 factor_hyperparameters = FACTOR_HYPERPARAMETERS
                 ):
        self.data = data
        self.data_depth = data_depth
        self.factor_hyperparameters = factor_hyperparameters

        self.factor_registry = {
            'obi_weighted': self.factor_obi_weighted,
            'spread': self.factor_spread,
            'spread_deviation': self.factor_spread_deviation,
            'ask_px_momentum': self.factor_ask_px_momentum,
            'bid_px_momentum': self.factor_bid_px_momentum,
            'vwap_mid_deviation_ask': self.factor_vwap_mid_deviation_ask,
            'vwap_mid_deviation_bid': self.factor_vwap_mid_deviation_bid,
        }

    def generate_factors(self,
                         factor_list = None) -> pd.DataFrame:
        factors = []

        if factor_list is None:
            factor_list = list(self.factor_hyperparameters.keys())
        for factor_name in factor_list:
            factors.append(self.factor_registry[factor_name]())
        return pd.concat(factors, axis=1)


    def factor_obi_weighted(self,
                            level_weights = None) ->pd.DataFrame:
        """
        Return dataframe with weighted obi factors, with weights in each level
        specified by level_weights; len(level_weights) == self.data_depth
        """
        factor_name = 'obi_weighted'
        if level_weights is None:
            level_weights = self.factor_hyperparameters[factor_name]["level_weights"]

        assert len(level_weights) == self.data_depth
        # assert(sum(level_weights) == 100) # obi is scale invariant, would divide by total_size

        weighted_bid_sz = sum(level_weights[level] * self.data[f"bid_sz_{level:02d}"]
                           for level in range(self.data_depth))
        weighted_ask_sz = sum(level_weights[level] * self.data[f"ask_sz_{level:02d}"]
                            for level in range(self.data_depth))

        total_sz = weighted_bid_sz + weighted_ask_sz
        obi_weighted = (weighted_bid_sz-weighted_ask_sz) / total_sz

        assert len(obi_weighted) == self.data.shape[0]
        return obi_weighted.rename(factor_name).to_frame()

    def factor_spread(self):
        factor_name = 'spread'
        spread = self.data['ask_px_00'] - self.data['bid_px_00']
        return spread.rename(factor_name).to_frame()

    def factor_spread_deviation(self,
                                look_back_ticks = None):
        """
        spread deviation from volume-weighted average spread over past ticks
        (not including current tick)
        """
        factor_name = 'spread_deviation'
        if look_back_ticks is None:
            look_back_ticks = self.factor_hyperparameters[factor_name]["look_back_ticks"]

        spread = self.data['ask_px_00'] - self.data['bid_px_00']
        volume = self.data['bid_sz_00'] + self.data['ask_sz_00']

        rolling_sum_weighted_spread = (spread * volume).rolling(look_back_ticks, min_periods=look_back_ticks).sum()
        rolling_sum_volume = volume.rolling(look_back_ticks, min_periods=look_back_ticks).sum()
        average_spread = rolling_sum_weighted_spread / rolling_sum_volume

        spread_deviation = spread - average_spread.shift(1)
        return spread_deviation.rename(factor_name).to_frame()

    def _px_momentum(self, side, factor_name, lag_ticks):
        if lag_ticks is None:
            lag_ticks = self.factor_hyperparameters[factor_name]["lag_ticks"]

        px = self.data[f'{side}_px_00']
        momentum = np.log(px / px.shift(lag_ticks))
        return momentum.rename(factor_name).to_frame()

    def factor_ask_px_momentum(self,
                               lag_ticks = None):
        return self._px_momentum('ask', 'ask_px_momentum', lag_ticks)

    def factor_bid_px_momentum(self,
                               lag_ticks = None):
        return self._px_momentum('bid', 'bid_px_momentum', lag_ticks)

    def _vwap_mid_deviation(self,
                    side,
                    factor_name,
                    look_back_ticks = None,
                    level_weights = None,
                    ):
        """
        Log deviation of the rolling quote-book VWAP per side from the current
        mid price: ln(vwap / mid). The VWAP is the average displayed price over
        the past look_back_ticks book updates, weighted by displayed size at
        each level and by level_weights. Built from resting quotes, not trades.

        Expressing the VWAP relative to the current mid removes the
        nonstationary price level, leaving a roughly mean-0 signal: negative
        means the current mid is above its recent size-weighted average
        (price has drifted up), positive means below.

        Suggested:
        1. put a high weight on top levels — they carry far less resting
        volume than deep levels, so without upweighting the deep levels would
        dominate the average.
        2. Use level_weights = (100, 0, 0, ...) to get best bid/ask vwap

        """

        if look_back_ticks is None:
            look_back_ticks = self.factor_hyperparameters[factor_name]["look_back_ticks"]
        if level_weights is None:
            level_weights = self.factor_hyperparameters[factor_name]["level_weights"]
        assert len(level_weights) == self.data_depth

        def calculated_vwap():
            """
            Quote-book VWAP for one side over the past look_back_ticks:

                vwap_t = sum_{i=t-N+1..t} sum_{level} w_l * sz_{i,level} * px_{i,level}
                         -----------------------------------------------
                         sum_{i=t-N+1..t} sum_{level} w_l * sz_{i,level}

            i.e. the average displayed price, weighted by displayed size at
            each level and by level_weights (w_l), over the trailing window
            of N = look_back_ticks book updates. Uses resting quotes, not
            trades. First N-1 rows are NaN; window includes the current tick.
            """

            # level_weights weighted total size at each tick
            adjusted_sz= sum(self.data[f"{side}_sz_{level:02d}"] *
                             level_weights[level]
                             for level in range(self.data_depth))

            # level_weights and size weighted price at each tick
            adjusted_notional = sum(self.data[f"{side}_sz_{level:02d}"] *
                               self.data[f"{side}_px_{level:02d}"] *
                               level_weights[level]
                               for level in range(self.data_depth))


            rolling_sum_notional = adjusted_notional.rolling(look_back_ticks, min_periods=look_back_ticks).sum()
            rolling_sum_sz = adjusted_sz.rolling(look_back_ticks, min_periods=look_back_ticks).sum()
            vwap = rolling_sum_notional / rolling_sum_sz
            return vwap

        mid = (self.data['bid_px_00'] + self.data['ask_px_00']) / 2
        vwap_mid_deviation = np.log(calculated_vwap() / mid)
        return vwap_mid_deviation.rename(factor_name).to_frame()

    def factor_vwap_mid_deviation_ask(self,
                                      look_back_ticks = None,
                                      level_weights = None):
        return self._vwap_mid_deviation('ask', 'vwap_mid_deviation_ask',
                                        look_back_ticks, level_weights)

    def factor_vwap_mid_deviation_bid(self,
                                      look_back_ticks = None,
                                      level_weights = None):
        return self._vwap_mid_deviation('bid', 'vwap_mid_deviation_bid',
                                        look_back_ticks, level_weights)

LABEL_HYPERPARAMETERS = {
    "ask_return": {"look_forward_ticks": 100},
    "ask_return_max": {"look_forward_ticks": 100},
    "ask_return_min": {"look_forward_ticks": 100},
    "ask_return_mean": {"look_forward_ticks": 100},
    "ask_return_direction": {"look_forward_ticks": 100},
    "ask_return_std": {"look_forward_ticks": 100},
    "ask_return_log_std": {"look_forward_ticks": 100},
}


class Labels:
    def __init__(self,
                 data,
                 label_hyperparameters = LABEL_HYPERPARAMETERS,
                 ):
        # goal is to buy 1 share per minute at best ask price
        # setup different labels which we can use to time execution
        self.data = data
        self.label_hyperparameters = label_hyperparameters

        self.label_registry = {
            "ask_return": self.label_ask_return,
            "ask_return_max": self.label_ask_return_max,
            "ask_return_min": self.label_ask_return_min,
            "ask_return_mean": self.label_ask_return_mean,
            "ask_return_direction": self.label_ask_return_direction,
            "ask_return_std": self.label_ask_return_std,
            "ask_return_log_std": self.label_ask_return_log_std,
        }

        self.label_desc = {
            "ask_return": "Log return of the ask price after look_forward_ticks, calculated as "
                          "ln(ask_{t+H} / ask_t).",

            "ask_return_max": "Worst-case ask return of delaying the buy, calculated as "
                              "ln(max(ask_{t+1..t+H}) / ask_t): the highest ask over the "
                              "forecast horizon relative to the current ask. Represents the "
                              "risk of the ask price going up while waiting; more positive "
                              "means a bigger potential loss.",

            "ask_return_min": "Best-case ask return of delaying the buy, calculated as "
                              "ln(min(ask_{t+1..t+H}) / ask_t): the lowest ask over the "
                              "forecast horizon relative to the current ask. Represents the "
                              "maximum price improvement achievable by waiting with perfect "
                              "timing; more negative means a bigger potential saving.",

            "ask_return_mean": "Expected ask return of delaying the buy, calculated as "
                               "ln(mean(ask_{t+1..t+H}) / ask_t): the average ask over the "
                               "forecast horizon relative to the current ask. Negative means "
                               "waiting is expected to get a better price; positive means buy now.",

            "ask_return_direction": "Direction of the forward ask return, binary: 1 if the ask price "
                                    "is higher after look_forward_ticks (buy now), "
                                    "0 if it is unchanged or lower (waiting gets a better price).",

            "ask_return_std": "Volatility of the ask return over the forecast horizon, calculated as "
                              "std(ln(ask_{t+1..t+H} / ask_t)): the standard deviation of the log "
                              "returns of future asks relative to the current ask. Higher means more "
                              "uncertainty in the price achievable by waiting, beyond what "
                              "ask_return_mean predicts.",

            "ask_return_log_std": "Log of ask_return_std, the regression-friendly form of the "
                                  "volatility target: realized std is non-negative and right-skewed, "
                                  "so in levels a squared-error fit is dominated by the few most "
                                  "volatile windows and can predict negative std; in logs errors "
                                  "become multiplicative and a linear model cannot go negative.",
        }


    def generate_labels(self,
                        label_list = None):
        if label_list is None:
            label_list = self.label_registry.keys()

        labels = []
        for label_name in label_list:
            func = self.label_registry[label_name]
            labels.append(func())

        res = pd.concat(labels, axis = 1)
        return  res

    def label_ask_return(self,
                         look_forward_ticks = None,):
        label_name = 'ask_return'
        if look_forward_ticks is None:
            look_forward_ticks = self.label_hyperparameters[label_name]['look_forward_ticks']

        ask_return = np.log(self.data["ask_px_00"].shift(-1*look_forward_ticks) /
                            self.data["ask_px_00"])

        return ask_return.rename(label_name).to_frame()

    def label_ask_return_max(self,
                             look_forward_ticks = None,):
        label_name = 'ask_return_max'
        if look_forward_ticks is None:
            look_forward_ticks = self.label_hyperparameters[label_name]['look_forward_ticks']

        # max best ask price in futrue look_forward_ticks, excluding current tick
        max_ask = self.data['ask_px_00'].rolling(look_forward_ticks).max().shift(-1*look_forward_ticks)
        ask_return_max = np.log(max_ask / self.data["ask_px_00"])

        return ask_return_max.rename(label_name).to_frame()

    def label_ask_return_min(self,
                             look_forward_ticks = None,):
        label_name = 'ask_return_min'
        if look_forward_ticks is None:
            look_forward_ticks = self.label_hyperparameters[label_name]['look_forward_ticks']

        # min best ask price in future look_forward_ticks, excluding current tick
        min_ask = self.data['ask_px_00'].rolling(look_forward_ticks).min().shift(-1*look_forward_ticks)
        ask_return_min = np.log(min_ask / self.data["ask_px_00"])

        return ask_return_min.rename(label_name).to_frame()

    def label_ask_return_mean(self,
                              look_forward_ticks = None,):
        label_name = 'ask_return_mean'
        if look_forward_ticks is None:
            look_forward_ticks = self.label_hyperparameters[label_name]['look_forward_ticks']

        # mean (equal weighted) best ask price in future look_forward_ticks, excluding current tick;
        # if we uniformly choose a random tick in the window, this mean price is what we would pay in expectation
        mean_ask = self.data['ask_px_00'].rolling(look_forward_ticks).mean().shift(-1*look_forward_ticks)
        ask_return_mean = np.log(mean_ask / self.data["ask_px_00"])
        return ask_return_mean.rename(label_name).to_frame()


    def label_ask_return_direction(self,
                                   look_forward_ticks = None,):
        label_name = 'ask_return_direction'
        if look_forward_ticks is None:
            look_forward_ticks = self.label_hyperparameters[label_name]['look_forward_ticks']

        # 1: forward ask > current ask, buy now
        # 0: forward ask <= current ask, wait
        forward_ask_px = self.data["ask_px_00"].shift(-look_forward_ticks)
        ask_return_direction = (forward_ask_px > self.data["ask_px_00"]).astype(float)

        # nan > x will be False, which would be assigned to 0 in label
        # the following lines manually adjust these rows back to nan
        nan_mask = forward_ask_px.isna() | self.data["ask_px_00"].isna()
        ask_return_direction[nan_mask] = np.nan

        return ask_return_direction.rename(label_name).to_frame()

    def label_ask_return_std(self,
                             look_forward_ticks = None,):
        label_name = 'ask_return_std'
        if look_forward_ticks is None:
            look_forward_ticks = self.label_hyperparameters[label_name]['look_forward_ticks']

        # std is shift-invariant: std(ln(ask_{t+1..t+H} / ask_t)) = std(ln(ask_{t+1..t+H})),
        # so we can take a rolling std of log prices instead of pairwise returns
        log_ask = np.log(self.data["ask_px_00"])
        ask_return_std = log_ask.rolling(look_forward_ticks).std().shift(-look_forward_ticks)

        return ask_return_std.rename(label_name).to_frame()

    def label_ask_return_log_std(self,
                                 look_forward_ticks = None,):
        label_name = 'ask_return_log_std'
        if look_forward_ticks is None:
            look_forward_ticks = self.label_hyperparameters[label_name]['look_forward_ticks']

        ask_return_std = self.label_ask_return_std(look_forward_ticks)['ask_return_std']

        # Zero-vol windows (ask unchanged for the whole horizon) would give
        # log(0) = -inf. rolling().std() also returns float noise (~1e-10)
        # instead of exact 0 for many unchanged windows, far below the
        # smallest real std (~1e-6, one tick move in the window), so treat
        # everything below NOISE_EPS as zero-vol and clip both cases to the
        # smallest meaningful observed std.
        NOISE_EPS = 1e-10
        floor = ask_return_std[ask_return_std > NOISE_EPS].min()
        ask_return_log_std = np.log(ask_return_std.clip(lower=floor))

        return ask_return_log_std.rename(label_name).to_frame()


def grid_search_factors(data,
                        factor_grid,
                        look_forward_ticks,
                        resample_freq = "1s", # can be None
                        data_depth = 3):
    """
    Grid search over factor hyperparameters, scored by correlation against
    all labels computed with a single shared look_forward_ticks.

    factor_grid maps factor name -> {param: list of candidate values}, e.g.
        {'spread': {'look_back_ticks': [50, 100, 300]}}

    Returns a correlation DataFrame with one row per factor-column variant
    (named 'column|param=value,...') and one column per label.
    """
    label_hyperparameters = {name: {'look_forward_ticks': look_forward_ticks}
                             for name in LABEL_HYPERPARAMETERS}
    labels = Labels(data, label_hyperparameters).generate_labels()

    factors_generator = Factors(data, data_depth=data_depth)
    factor_variants = {}
    for factor_name, grid in factor_grid.items():
        # grid: dict[param_name, list_of_param_values],
        # e.g. {'look_back_ticks': [100, 300],
        #       'level_weights': [(100,0,0), (85,10,5)]
        #       }

        param_names = list(grid.keys())
        for values in product(*grid.values()):
            # "*" unpacks grid.values(); values is a tuple

            params = dict(zip(param_names, values))
            # e.g. params = {'look_back_ticks': 30, 'level_weights': (100,0,0)}

            tag = ','.join(f'{param_name}={value}' for param_name, value in params.items())
            factor_df = factors_generator.factor_registry[factor_name](**params)
            for col in factor_df.columns:
                factor_variants[f'{col}|{tag}'] = factor_df[col]

    factor_variants = pd.DataFrame(factor_variants)

    data = pd.concat([factor_variants, labels], axis=1)
    if resample_freq:
        data = data.resample(resample_freq).first().ffill()

    corr = data.corr()
    return corr.loc[factor_variants.columns, labels.columns]


def load_factors_and_labels(
        data,
        data_depth=3,
        factor_hyperparameters = FACTOR_HYPERPARAMETERS,
        label_hyperparameters = LABEL_HYPERPARAMETERS,
        dropna = True,):

    list_factor_names = list(factor_hyperparameters.keys())
    list_label_names = list(label_hyperparameters.keys())

    df_factors = (Factors(data, data_depth, factor_hyperparameters)
                  .generate_factors(factor_list=list_factor_names))
    df_labels = (Labels(data, label_hyperparameters)
                 .generate_labels(label_list=list_label_names))
    if dropna:
        mask_na = df_factors.isna().any(axis=1) | df_labels.isna().any(axis=1)
        df_factors = df_factors.loc[~mask_na, :]
        df_labels = df_labels.loc[~mask_na, :]
        print(f"Dropped {sum(mask_na)} rows with NaN values")

    return df_factors, df_labels