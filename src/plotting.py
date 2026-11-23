import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _save(save_path) -> None:
    """Save the current figure (must be called before plt.show(), which clears it)."""
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)


def plot_normalized_prices(prices: pd.DataFrame, save_path=None) -> None:

    normalized_prices = prices / prices.iloc[0]

    normalized_prices.plot(figsize=(12, 6))

    plt.title ("Normalized Price Development")
    plt.xlabel ("Date")
    plt.ylabel ("Normalized price")
    plt.grid (True)
    plt.tight_layout()
    _save(save_path)
    plt.show()


def plot_10y_yield(yield_series: pd.Series, save_path=None) -> None:

    yield_series.plot(figsize=(12, 6))

    plt.title ("10y Yield")
    plt.xlabel ("Date")
    plt.ylabel ("10y Yield")
    plt.grid (True)
    plt.tight_layout()
    _save(save_path)
    plt.show()



def plot_interest_rate_shocks(yield_changes_bsp: pd.Series, shock_mask: pd.Series) -> None:

    yield_changes_bsp, shock_mask = yield_changes_bsp.align(shock_mask, join = "inner")

    plt.figure(figsize = (12, 6))

    plt.plot(yield_changes_bsp.index, yield_changes_bsp, label = "Yield Change")
    plt.scatter(yield_changes_bsp[shock_mask].index, yield_changes_bsp[shock_mask], label = "Shock days")

    plt.title("Interest Shock Days")
    plt.xlabel("Date")
    plt.ylabel("Yield Change In Basis Points")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_yield_with_shock_days(yield_10y, shock_days, lag_days=0, save_path=None):

    plt.figure(figsize=(12, 5))

    plt.plot(yield_10y.index, yield_10y, label="10Y Yield")

    # Alle Tage markieren (Schocktag + Lags)
    marked_dates = set()

    for date in shock_days.index:
        loc = yield_10y.index.get_loc(date)

        for lag in range(lag_days + 1):  # 0 = Schocktag
            if loc + lag < len(yield_10y.index):
                marked_dates.add(yield_10y.index[loc + lag])

    marked_dates = sorted(marked_dates)

    plt.scatter(
        marked_dates,
        yield_10y.loc[marked_dates],
        color="red",
        s=0.5,
        label=f"Shock Days (+{lag_days} Lags)",
        zorder=3
    )

    plt.title("10-Year Treasury Yield with Interest Rate Shock Periods")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _save(save_path)
    plt.show()



def plot_average_returns_on_shock_days(shock_returns:pd.Series) -> None:

    shock_returns.plot(kind = "bar", figsize =(8,5))

    plt.title("Average Returns During Interest Shocks")
    plt.xlabel("Asset")
    plt.ylabel("Average Log Return")
    plt.grid(True, axis = "y")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(data: pd.DataFrame, title = None) -> None:

    correlation_matrix = data.corr()

    plt.figure(figsize = (8, 5))
    plt.imshow(correlation_matrix)

    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns) # Letzter Eintrag ist Name der Spalten 
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)

    plt.colorbar(label = "Correlation")
    if title:
        plt.title(title)

    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, round(correlation_matrix.iloc[i, j], 2), ha = "center", va = "center")
    
    plt.tight_layout()
    


def plot_portfolio_performance(portfolio_returns: pd.DataFrame, asset_returns: pd.DataFrame | None = None) -> None:

    portfolio_performance = np.exp(portfolio_returns.cumsum())

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_performance.index, portfolio_performance, label = "Portfolio")

    if asset_returns is not None:
        asset_performance = np.exp(asset_returns.cumsum())

        for column in asset_performance.columns:
            plt.plot(asset_performance.index, asset_performance[column], label = column)

    plt.title("Portfolio Performance")
    plt.xlabel("Date")
    plt.ylabel("Growth of 1 Unit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_portfolio_drawdown(portfolio_returns: pd.DataFrame) -> None:

    cummulative_performance = np.exp(portfolio_returns.cumsum())
    running_max = cummulative_performance.cummax()
    drawdown = cummulative_performance / running_max - 1

    drawdown.plot(figsize = (12, 5))

    plt.title("Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_sharpe_vs_tlt_weight(optimization_results: pd.DataFrame) -> None:

    plt.figure(figsize=(10, 5))

    plt.plot(optimization_results["TLT Weight"],
        optimization_results["Sharpe Ratio"],
        marker="o"
    )

    plt.title("Sharpe Ratio vs. TLT Weight")
    plt.xlabel("TLT Weight")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_volatility_vs_tlt_weight(optimization_results: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))

    plt.plot(
        optimization_results["TLT Weight"],
        optimization_results["Annual Volatility"],
        marker="o"
    )

    plt.title("Portfolio Volatility vs. TLT Weight")
    plt.xlabel("TLT Weight")
    plt.ylabel("Annualized Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_drawdown_vs_tlt_weight(optimization_results: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))

    plt.plot(
        optimization_results["TLT Weight"],
        optimization_results["Max Drawdown"],
        marker="o"
    )

    plt.title("Maximum Drawdown vs. TLT Weight")
    plt.xlabel("TLT Weight")
    plt.ylabel("Maximum Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_sharpe_surface_heatmap(sharpe_matrix: pd.DataFrame, title: str = "Sharpe Ratio Surface") -> None:

    plt.figure(figsize=(10, 7))

    plt.imshow(sharpe_matrix, origin = "lower", aspect = "auto")
    plt.colorbar(label="Sharpe Ratio")

    plt.xticks(range(len(sharpe_matrix.columns)), sharpe_matrix.columns)

    plt.yticks(range(len(sharpe_matrix.index)), sharpe_matrix.index)

    plt.title(title)
    plt.xlabel("NVDA Weight")
    plt.ylabel("TLT Weight")
    plt.tight_layout()
    plt.show()


def plot_sharpe_surface_3d(
    sharpe_matrix: pd.DataFrame,
    title: str = "3D Sharpe Ratio Surface"
) -> None:

    from mpl_toolkits.mplot3d import Axes3D

    x_values = sharpe_matrix.columns.astype(float)
    y_values = sharpe_matrix.index.astype(float)

    X, Y = np.meshgrid(x_values, y_values)
    Z = sharpe_matrix.values

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(X, Y, Z, cmap="viridis")

    fig.colorbar(surface, ax=ax, label="Sharpe Ratio")

    ax.set_title(title)
    ax.set_xlabel("NVDA Weight")
    ax.set_ylabel("TLT Weight")
    ax.set_zlabel("Sharpe Ratio")

    plt.tight_layout()
    plt.show()

def plot_sharpe_ratio_by_lag(optimal_portfolios: pd.DataFrame) -> None:

    plt.figure(figsize=(8,5))

    plt.plot(optimal_portfolios.index, optimal_portfolios["Sharpe Ratio"], marker="o")

    best_lag = optimal_portfolios["Sharpe Ratio"].idxmax()
    best_sharpe = optimal_portfolios.loc[best_lag, "Sharpe Ratio"]

    plt.scatter(best_lag, best_sharpe, s=100, label=f"Best Lag: {best_lag}")

    plt.title("Optimal Sharpe Ratio Depending on Lag")
    plt.xlabel("Lag after Interest Rate Shock")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_optimal_weights_by_lag(optimal_portfolios: pd.DataFrame) -> None:

    optimal_portfolios[["NVDA Weight", "XLI Weight", "TLT Weight"]].plot(figsize=(10,6), marker="o")

    plt.title("Optimal Portfolio Composition by Lag")
    plt.xlabel("Lag after Interest Rate Shock")
    plt.ylabel("Portfolio Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.show()