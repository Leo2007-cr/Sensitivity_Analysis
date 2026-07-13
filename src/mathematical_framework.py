import pandas as pd
import numpy as np

def get_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Return cleaned dataprice frame
    """
    return prices.dropna()


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:

    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def calculate_yield_change_bps(yields: pd.DataFrame) -> pd.DataFrame:

    yield_change_bps = yields.diff() * 100
    return yield_change_bps


def detect_rate_shocks(yield_change_bps: pd.Series, quantile: float = 0.90) -> pd.Series:

    threshold = yield_change_bps.abs().quantile(quantile)
    shocks = yield_change_bps.abs() >= threshold
    return shocks


def average_returns_during_shocks (asset_returns: pd.DataFrame, shock_mask: pd.Series) -> pd.Series:

    aligned_returns, aligned_shocks = asset_returns.align(shock_mask, join = "inner", axis = 0)
    return aligned_returns[aligned_shocks].mean()


def relative_interest_rate_sensitivity (returns: pd.DataFrame, asset_1: str = "NVDA", asset_2: str = "XLI") -> pd.Series:

    """
    Positive means asset_1 outperformed asset_2 and vice versa for negative values
    """
    relative_returns = returns[asset_1] - returns[asset_2]
    return relative_returns.dropna()


def correlation_with_yield_changes (asset_returns: pd.DataFrame, yield_changes_bps: pd. Series) -> pd.Series:

    aligned_returns, aligned_yields = asset_returns.align(yield_changes_bps, join = "inner", axis = 0)
    return aligned_returns.corrwith(aligned_yields)


def calculate_portfolio_returns (returns: pd.DataFrame, weights: pd.Series) -> pd.Series:

    weights_series = pd.Series(weights)
    returns = returns[weights_series.index]
    portfolio_returns = returns.mul(weights_series, axis = 1).sum(axis = 1)
    return portfolio_returns


def calculate_portfolio_volatility(returns: pd.DataFrame, weights: pd.Series, annulazied: bool = True) -> float:

    weights_series = pd.Series(weights)
    returns = returns[weights_series.index]
    covariance_matrix = returns.cov()
    weights_array = weights_series.values
    portfolio_variance = weights_array.T @ covariance_matrix @ weights_array
    portfolio_volatility = np.sqrt(portfolio_variance)
    if annulazied:
        portfolio_volatility *= np.sqrt(252)
    return portfolio_volatility


def optimize_tlt_weight(returns: pd.DataFrame, nvda_weight: float, xli_weight: float, tlt_range: np.ndarray = np.arrange(0, 0.55, 0.05)) -> pd.DataFrame:

    results = []
    equity_total = nvda_weight + xli_weight

    for tlt_weight in tlt_range:
        remaining_weight = 1 - tlt_weight
        adjusted_nvda_weight = remaining_weight * (nvda_weight / equity_total)
        adjusted_xli_weight = remaining_weight * (xli_weight / equity_total)
        # Increasing TLT requires symmetric decreasing in XLI, NVDA with the half of the increase with respect wo NVDA, XLI

        weights = {"NVDA": adjusted_nvda_weight, "XLI": adjusted_xli_weight, "TLT": tlt_weight}

        portfolio_returns = calculate_portfolio_returns(returns, weights)
        annual_return = portfolio_returns.mean() * 252
        annual_volatilty = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatilty

        results.append({"TLT Weight": tlt_weight, "NVDA Weight": adjusted_nvda_weight, "XLI Weight": xli_weight, 
                        "Annual Return": annual_return, "Annual Volatility": annual_volatilty, "Annual Ratio": sharpe_ratio})
        
    return pd.DataFrame(results)
        

def calculate_performance_metrices(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:

    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility # in notebook using different values for risk free rate to be more realistic

    cumulative_return = np.exp(returns.sum())-1

    max_drawdown = calculate_max_drawdown(returns)
    return {"Annual Return": annual_return, 
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Cumulative Return": cumulative_return,
        "Max Drawdown": max_drawdown}
 

def calculate_max_drawdown(returns: pd.Series) -> float:
    
    cumulative_returns = np.exp(returns.cumsum())
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1

    return drawdown.min()