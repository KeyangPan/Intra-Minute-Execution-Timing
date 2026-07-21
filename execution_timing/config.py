# Tuned hyperparameters, shared across notebooks.

# Direction and volatility models use different lookbacks: the direction
# signal lives at short lags, vol prefers long ones.
FACTOR_HYPERPARAMETERS_DIRECTION = {
    'obi_weighted': {'level_weights': (80, 20, 0)},
    # 'spread': {},
    'spread_deviation': {'look_back_ticks': 300},
    'ask_px_momentum': {'lag_ticks': 50},
    'bid_px_momentum': {'lag_ticks': 50},
    'vwap_mid_deviation_ask': {'look_back_ticks': 30,
                               'level_weights': (100, 0, 0)},
    # 'vwap_mid_deviation_bid': {'look_back_ticks': 30,
    #                            'level_weights': (100, 0, 0)},
}

FACTOR_HYPERPARAMETERS_VOLATILITY = {
    'obi_weighted': {'level_weights': (34, 33, 33)},
    'spread': {},
    'spread_deviation': {'look_back_ticks': 600},
    'ask_px_momentum': {'lag_ticks': 600},
    'bid_px_momentum': {'lag_ticks': 600},
    'vwap_mid_deviation_ask': {'look_back_ticks': 30,
                               'level_weights': (100, 0, 0)},
    'vwap_mid_deviation_bid': {'look_back_ticks': 30,
                               'level_weights': (100, 0, 0)},
}

# Shared forecast horizon; defines the prediction problem
LOOK_FORWARD_TICKS = 300
LABEL_HYPERPARAMETERS = {
    'ask_return_direction': {'look_forward_ticks': LOOK_FORWARD_TICKS},
    # 'ask_return_log_std': {'look_forward_ticks': LOOK_FORWARD_TICKS},
}
