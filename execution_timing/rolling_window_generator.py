from dataclasses import dataclass
import pandas as pd


@dataclass
class Window:
    X_train: pd.DataFrame
    y_train: pd.DataFrame | pd.Series
    X_test: pd.DataFrame
    y_test: pd.DataFrame | pd.Series

    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class RollingWindowGenerator:
    """
    Builds successive (train, test) Windows over time-indexed X/y:

        train = [start, start + train_size)
        test  = [start + train_size, start + train_size + test_size)

    then start moves forward by step_size and repeats until the test
    window no longer fits inside the data.

    purge_secs drops the tail of each train window so that forward-looking
    labels (look_forward_ticks ahead) cannot peek into the test window.
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.DataFrame | pd.Series,
                 train_window_size_mins,
                 test_window_size_mins,
                 step_size_mins,
                 purge_secs=1,
                 ):
        self.X = X
        self.y = y
        self.train_window_size_mins = train_window_size_mins
        self.test_window_size_mins = test_window_size_mins
        self.step_size_mins = step_size_mins
        self.purge_secs = purge_secs

    def __iter__(self):
        train_size = pd.Timedelta(minutes=self.train_window_size_mins)
        test_size = pd.Timedelta(minutes=self.test_window_size_mins)
        step_size = pd.Timedelta(minutes=self.step_size_mins)
        purge = pd.Timedelta(seconds=self.purge_secs)

        index = self.X.index
        start = index[0].floor('min')
        end = index[-1].ceil('min')

        while start + train_size + test_size <= end:
            train_end = start + train_size
            test_end = train_end + test_size

            train_mask = (index >= start) & (index < train_end - purge)
            test_mask = (index >= train_end) & (index < test_end)

            yield Window(
                X_train=self.X[train_mask],
                y_train=self.y[train_mask],
                X_test=self.X[test_mask],
                y_test=self.y[test_mask],
                train_start=start,
                test_start=train_end,
                test_end=test_end,
            )
            start += step_size
