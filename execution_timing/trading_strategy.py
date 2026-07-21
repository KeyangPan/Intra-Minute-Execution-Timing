import numpy as np
import pandas as pd
from execution_timing.rolling_window_generator import Window
from execution_timing.prediction_model import PredictionModel

def signal_resample(sig, resample_freq):
    """
    Resample signal into lower frequency to make trading behavior more stable, will
    only select first signal during each time frequency bin
    """
    sig_df = sig.to_frame()
    sig_df['ts_event'] = sig_df.index
    sig_df = sig_df.resample(resample_freq).first().dropna(axis=0).set_index('ts_event')
    return sig_df[sig.name].astype('int')



class StrategyDirection:

    def __init__(self,
                 dir_pred_model: PredictionModel,
                 dir_threshold: float,
                 sig_resample_freq = None):
        self.model = dir_pred_model
        self.dir_threshold = dir_threshold
        self.sig_resample_freq = sig_resample_freq

    def predict(self,
               window: Window,
               ):
        """
        Train on window.X_train, predict on both train and test.
        """
        self.model.train(window.X_train, window.y_train)
        y_pred = self.model.predict_proba(window.X_test)
        y_pred_train = self.model.predict_proba(window.X_train)
        return y_pred, y_pred_train

    def generate_signal(self,
                        dir_pred,
                        dir_pred_train
                        ):
        # 1: predicted price goes up
        # 0: predicted price goes down
        # dir_threshold = dir_pred_train.iloc[-10000:].quantile(0.95) # 95% quantile
        mask_dir = dir_pred >= self.dir_threshold # price goes up in the future, favor buying now
        sig = (mask_dir).astype(int)
        sig.name = "signal"
        if self.sig_resample_freq is not None:
            sig = signal_resample(sig, self.sig_resample_freq)
        return sig

    def simulate(self,
                 window: Window,
                 ):
        y_pred, y_pred_train = self.predict(window)
        return self.generate_signal(y_pred, y_pred_train)

#
# class StrategyDirVol:
#     def __init__(self,
#                  dir_pred,
#                  vol_pred,
#                  dir_threshold,
#                  vol_threshold,
#                  sig_resample_freq=None
#                  ):
#         self.dir_pred = dir_pred
#         self.dir_threshold = dir_threshold
#
#         self.vol_pred = vol_pred
#         self.vol_threshold = vol_threshold
#
#         self.sig_resample_freq = sig_resample_freq
#         return
#
#     def generate_signal(self):
#         mask_vol = self.vol_pred >= self.vol_threshold  # large predicted vol
#         mask_dir = self.dir_pred >= self.dir_threshold  # best ask move up
#
#         sig = (mask_vol & mask_dir).astype(int)  # 1: Trade now to avoid future upmove
#         sig.name = "signal"
#
#         if self.sig_resample_freq is not None:
#             sig = signal_resample(sig, self.sig_resample_freq)
#         return sig
