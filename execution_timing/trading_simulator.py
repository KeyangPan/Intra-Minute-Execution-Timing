import numpy as np
import pandas as pd
from execution_timing.rolling_window_generator import Window
from collections.abc import Iterator
import pandas as pd


def run_simulation(strategy,
                   trading_simulator: TradingSimulator,
                   windows: Iterator[Window],):
    trading_result = []
    model_coeff = []
    for idx, window in enumerate(windows):
        signal = strategy.simulate(window)
        df = trading_simulator.run(signal)
        df.insert(0, '#window', idx)
        trading_result.append(df)
        coeff = strategy.model.get_model_coeff()
        coeff.name = f'window_{idx}'
        model_coeff.append(coeff)

    trading_result = pd.concat(trading_result, axis=0)
    model_coeff = pd.concat(model_coeff, axis=1)
    return trading_result, model_coeff


def summarize_results(sim_result: pd.DataFrame,):
    """
    Summary metrics for a TradingSimulator.run() output.
    """
    pnl = sim_result['pnl']
    stats = {
        'cumulative_pnl': pnl.sum(),
        'win_rate': (pnl > 0).mean(),
        'sharpe_ratio': pnl.mean() / pnl.std(),
        'mean_pnl': pnl.mean(),
        'mean_pnl_pct_spread': sim_result['pnl_pct_spread'].mean(),
        'trigger_rate': (sim_result['execution_second'] <= 59.5).mean(),
        'mean_arrival_spread': sim_result['arrival_spread'].mean(),
    }

    return pd.DataFrame([stats], index=['stats'])

def baseline_comparison(trading_result):
    # Naive timing baselines on the same minutes, from the reference columns
    # trading_simulator already records. Expected fill of a uniformly random
    # tick = the minute's mean ask, so pnl vs arrival = arrival - mean.
    baselines = pd.DataFrame({
        'buy_immediately': 0.0,
        'random_tick': trading_result['arrival_ask'] - trading_result['mean_ask'],
        'always_wait': trading_result['arrival_ask'] - trading_result['departure_ask'],
        'model_timed': trading_result['pnl'],
        'perfect_foresight': trading_result['arrival_ask'] - trading_result['min_ask'],
    })
    return baselines.mean().rename('mean_pnl_per_share').to_frame()

class TradingSimulator:
    """
    Executes a 0/1 signal under the task contract: buy exactly 1 share per
    minute at best ask.

    sig lives on the prediction grid (e.g. 0.5s); fills are looked up in the
    raw tick ask series, so decision resolution and price resolution stay
    separate.

    Decision rule per minute:
    - decide at the first sig == 1 grid point;
    - no signal in the minute -> decide at the minute's last grid point
      (deadline).

    Execution happens execution_lag_ticks RAW TICKS after the decision
    point (measured in ask_px, not on the sig grid): the signal is computed
    on the decision point's book, so the earliest realistic fill is a few
    ticks later. Lag 0 = fill at the book the signal saw (zero latency,
    optimistic bound).
    """

    def __init__(self,
                 raw_price: pd.DataFrame,
                 execution_lag_ticks: int = 0,
                 ):
        self.raw_px = raw_price
        self.execution_lag_ticks = execution_lag_ticks

    def get_decision_timestamp(self,
                               sig):
        """
        Decision point per minute, on the sig grid.

        Returns a Series: index = minute, value = decision timestamp
        (first sig == 1 grid point; no signal -> the minute's last grid
        point, i.e. the deadline).
        """
        index = sig.index
        minutes = index.floor('min')

        # integer position of each grid point
        pos = pd.Series(np.arange(len(index)), index=index)
        first_trigger_pos = pos[sig == 1].groupby(minutes[sig == 1]).first() # integer position of the first trigger point in each minute
        last_pos_of_minute = pos.groupby(minutes).last()

        decision_times = {}
        for minute, deadline_pos in last_pos_of_minute.items():
            # first signal of the minute; no signal -> the minute's last grid point
            if minute in first_trigger_pos.index:
                decision_pos = first_trigger_pos[minute]
            else:
                decision_pos = deadline_pos
            decision_times[minute] = index[decision_pos]

        return pd.Series(decision_times, name='decision_time')

    def get_execution_timestamp(self, sig):
        """
        Returns a Series: index = minute, value = execution timestamp
        (a raw-tick timestamp, execution_lag_ticks after the decision point).
        """
        decision_times = self.get_decision_timestamp(sig)
        tick_index = self.raw_px.index

        execution_times = {}
        for minute, decision_time in decision_times.items():
            # decision -> execution: lag in RAW ticks of ask_px (sig may be
            # resampled, ask_px is always original tick level).

            # searchsorted = binary search: position where decision_time
            # would be inserted; side='right' minus 1 = the last tick at or
            # before the decision, then step lag ticks forward.
            tick_pos = tick_index.searchsorted(decision_time, side='right') - 1
            tick_pos = min(tick_pos + self.execution_lag_ticks, len(tick_index) - 1)
            execution_times[minute] = tick_index[tick_pos]

        return pd.Series(execution_times, name='execution_time')

    def run(self, sig):
        """
        Simulate the strategy. Returns a per-minute DataFrame:

            execution_time   when the share was bought (raw-tick timestamp)
            execution_second seconds from minute start to execution
                             (~59.5+ = deadline fill at the minute's end)
            fill_price       ask at execution_time
            arrival_price    first ask of the minute ("buy immediately")
            mean_ask         mean ask over the minute (~ random-timing cost)
            min_ask          lowest ask of the minute (perfect foresight)
            pnl              arrival_price - fill_price, per share ($);
                             positive = waiting got a better price

        Reference prices are kept as columns (not pre-baked PnL) so any
        comparison -- savings vs arrival, capture ratio -- can be computed
        downstream.
        """
        execution_times = self.get_execution_timestamp(sig)

        # seconds from minute start to execution
        execution_second = ((execution_times - execution_times.index)
                            .dt.total_seconds()
                            .rename('execution_second'))

        # execution_time is an exact ask_px timestamp; take the last tick
        # at that timestamp (robust to duplicate timestamps)
        tick_index = self.raw_px.index
        fill_pos = tick_index.searchsorted(execution_times.array, side='right') - 1
        fill_price = pd.Series(self.raw_px['ask_px_00'].iloc[fill_pos].values,
                               index=execution_times.index, name='fill_price')

        res = pd.concat([execution_times, execution_second, fill_price], axis=1)

        # per-minute reference prices from the raw ticks
        def price_refs(group):
            return pd.Series({
                'arrival_ask': group['ask_px_00'].iloc[0],
                'departure_ask': group['ask_px_00'].iloc[-1],
                'mean_ask': group['ask_px_00'].mean(),
                'max_ask': group['ask_px_00'].max(),
                'min_ask': group['ask_px_00'].min(),
                'arrival_bid': group['bid_px_00'].iloc[0],
                'departure_bid': group['bid_px_00'].iloc[-1],
                # 'mean_bid': group['bid_px_00'].mean(),
                # 'min_bid': group['bid_px_00'].min(),
                'arrival_spread': group['ask_px_00'].iloc[0] - group['bid_px_00'].iloc[0],
                'departure_spread': group['ask_px_00'].iloc[-1] - group['bid_px_00'].iloc[-1],
            })

        refs = (self.raw_px[['ask_px_00', 'bid_px_00']]
                .groupby(tick_index.floor('min'))
                .apply(price_refs))

        res = res.join(refs, how='left') # join on index

        # per-share saving vs buying immediately at the minute open;
        # positive = waiting got a better price
        res['pnl'] = res['arrival_ask'] - res['fill_price']
        res['pnl_pct_spread'] = res['pnl'] / res['arrival_spread']
        return res
