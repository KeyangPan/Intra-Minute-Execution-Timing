import pandas as pd
import numpy as np

def read_raw_data(path: str,
                  trim_open_and_close: bool = True) -> pd.DataFrame:
    """
    Load a Databento parquet file, restricted to regular trading hours (ET).
    """
    df = pd.read_parquet(path)

    # Size/count columns are unsigned (uint32). Cast to signed int64 to avoid
    # underflow/wraparound on differences. Match every depth level.
    int_cols = [
        c for c in df.columns
        if c.startswith(("bid_sz_", "ask_sz_", "bid_ct_", "ask_ct_"))
    ]
    if "size" in df.columns:
        int_cols.append("size")
    df[int_cols] = df[int_cols].astype("int64")

    df["ts_event"] = pd.to_datetime(df["ts_event"]).dt.tz_convert("America/New_York")
    if trim_open_and_close:
        start_time, end_time = "10:00", "15:30"
    else:
        start_time, end_time = "19:30", "16:00"
    df = df.set_index("ts_event").between_time(start_time, end_time)
    return df

def truncate_depth(df_mbp_10: pd.DataFrame,
                   depth: int = 3) -> pd.DataFrame:
    """
    Truncate an MBP-10 dataframe to `depth` levels, dropping rows where the
    truncated book is unchanged.
    """
    book_cols = [
        f"{side}_{field}_{i:02d}"
        for field in ("px", "sz")
            for side in ("bid", "ask")
                for i in range(depth)]
    df_truncated = df_mbp_10.loc[:, book_cols]

    # Keep the first row of duplicate rows
    df_tmp = df_truncated.fillna(-1)  # sentinel so empty levels (NaN) compare equal
    mask = df_tmp.ne(df_tmp.shift()).any(axis=1)
    df_truncated = df_truncated.loc[mask]

    return df_truncated


def holdout_split(data: pd.DataFrame,
                  holdout_mins: int,
                  purge_ticks: int=0,
                  ):
    """
    Hold out last holdout_mins data.

    data: can be raw_data(MBP) or raw_data with factors and labels
    """
    cutoff_time = data.index[-1].ceil('min') - pd.Timedelta(minutes=holdout_mins)
    data_train = data[data.index < cutoff_time]
    data_holdout = data[data.index >= cutoff_time]

    if purge_ticks > 0:
        # prevent labels in train looking forward into holdout
            data_train = data_train.iloc[:-purge_ticks]

    return data_train, data_holdout
