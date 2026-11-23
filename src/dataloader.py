import pandas as pd
import yfinance as yf

# Tickers:
EQITY_TICKERS = ["NVDA", "XLI"] # Nvidia-Aktie- Preis und Industieunternehmen aus S&P500
BOND_TICKER = "TLT" # ETF auf langfrisitge US-Treasuries (20 Jahre und handelbar)
YIELD_TICKER = "^TNX" # US-Staatsanleihenrendite (10 Jahre und nicht handelbar)
PORTFOLIO_TICKERS = ["TLT", "NVDA", "XLI"]
ALL_TICKERS = ["NVDA", "TLT", "XLI", "^TNX"]

def download_yahoo_data(tickers, start = "2019-01-01", end = None):
    """
    Download adjusted closing data from Yahoo Finance.
    Parameters:
    tickers : list[str] or str
        Yahoo Finance tickers.
    start : str
        Start date in YYYY-MM-DD format.
    end : str or None
        End date in YYYY-MM-DD format.
    Returns:
    pd.DataFrame
        Adjusted close prices or yield levels.
    """

    data = yf.download(tickers = tickers, start = start, end = end, auto_adjust = True, progress = False) 
    # AutoAdjust = True, damit keine Crashs kommen bei Dividendenzahlung oder Aktiensplit
    # progress = False, kein Output/ Download-Fortschritt-Balken während Download
    # tickers = tickers, so im Notebook ALL oder nur bestimmte Ticker auswählbar

    if data. empty:
        raise ValueError(" No Data was downloaded")
    close = data ["Close"]
    if isinstance(close, pd.Series): # Einfache Spalte zu Spalte mit Einträgen plus Indexspalte
        close = close.to_frame()
    close = close.dropna(how="all")
    return close

def load_equity_data(start = "2019-01-01", end = None):
    """
    Load NVIDIA and industrial equity data: NVDA and XLI
    """
    return download_yahoo_data(tickers = EQITY_TICKERS, start = start, end = end)

def load_portfolio_assets(start = "2019-01-01", end = None):
    """
    Load tradable portfolio assets:
    NVDA, XLI and TLT.
    """
    return download_yahoo_data(tickers = PORTFOLIO_TICKERS, start = start, end = end)

def load_10y_yield(start = "2019-01-01", end = None):
    """
    Load the 10-Year Treasury Yield.
    Yahoo Finance ticker:
    ^TNX
    Note:
    ^TNX is quoted in yield points.
    A value of 4.25 corresponds approximately to 4.25%.
    """
    yield_data = download_yahoo_data(tickers = YIELD_TICKER, start = start, end = end)
   
    yield_data.columns = ["10Y_Yield"]
    
    return yield_data

def load_all_project_data(start = "2019-01-01", end = None):
    """
    Load all time series required for the project:
    NVDA, XLI, TLT and 10-Year Treasury Yield.
    """
    data = download_yahoo_data(tickers = ALL_TICKERS, start = start, end = end)
    data = data.rename(columns={"^TNX": "10Y_Yield"})
    return data.dropna()

def align_data(*dataframes):
    """
    Align several dataframes on common dates.
    """
    return pd.concat(
        dataframes,
        axis=1
    ).dropna()

def split_project_data(full_data):
    """
    Split full project data into assets and yield series.
    Returns
    portfolio_assets : pd.DataFrame
        NVDA, XLI and TLT price data.
    yield_10y : pd.DataFrame
        10-Year Treasury Yield.
    """
    portfolio_assets = full_data[["XLI", "TLT", "NVDA"]]
    yield_10y = full_data [["10Y_Yield"]]
    return portfolio_assets, yield_10y

def save_to_csv(dataframe, path):
    """
    Save dataframe to CSV.
    """
    dataframe.to_csv(path)

def load_from_csv(path):
    """
    Load dataframe from local CSV file.
    """
    return pd.read_csv(path, index_col=0,parse_dates=True)