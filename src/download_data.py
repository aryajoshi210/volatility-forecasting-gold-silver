"""Downloads daily gold and silver futures prices from Yahoo Finance and saves them to data/."""

import yfinance as yf

TICKERS = {
    "gold": "GC=F",
    "silver": "SI=F",
}

START_DATE = "2010-01-01"


def download_and_save(name, ticker):
    print(f"Downloading {name} ({ticker})...")
    data = yf.download(ticker, start=START_DATE)
    output_path = f"data/{name}_futures.csv"
    data.to_csv(output_path)
    print(f"Saved {len(data)} rows to {output_path}")


if __name__ == "__main__":
    for name, ticker in TICKERS.items():
        download_and_save(name, ticker)
