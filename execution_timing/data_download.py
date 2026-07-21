import databento as db
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()
databento_api_key = os.getenv("DATABENTO_API_KEY")

def download_stock_data(symbol:str,
                        date:str,
                        schema:str = 'mbp-10'):
    client = db.Historical(key=databento_api_key)

    start = pd.Timestamp(f"{date} 09:30", tz="America/New_York")
    end = pd.Timestamp(f"{date} 16:00", tz="America/New_York")

    data = client.timeseries.get_range(
        dataset="XNAS.ITCH",
        schema=schema,
        symbols=[symbol],
        start=start,
        end=end,
        # path=f"data/{symbol.lower()}_mbp10_{date}.dbn.zst",  # stream to disk instead of RAM
    )

    data.to_parquet(f"data_raw/{symbol.lower()}_{schema}_{date}.parquet")

download_stock_data('NVDA',
                    '2026-07-17',
                    'mbp-10')