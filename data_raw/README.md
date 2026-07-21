# data_raw

Raw market data used by the notebooks. Parquet files are not committed
(see `.gitignore`) because Databento data cannot be redistributed.

Expected contents:

```
nvda_mbp-10_2026-07-01.parquet
nvda_mbp-10_2026-07-02.parquet
nvda_mbp-10_2026-07-17.parquet
```

To download, put your Databento API key in a `.env` file at the project root:

```
DATABENTO_API_KEY=<your key>
```

then run `hft_timing/data_download.py`, which fetches NVDA MBP-10 data
(Nasdaq XNAS.ITCH, regular trading hours) via `download_stock_data(symbol, date)`
and writes the parquet files here.
