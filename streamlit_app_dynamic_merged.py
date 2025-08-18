import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from backtest import run_backtest_simulation  # Make sure backtest.py is present and imported

st.set_page_config(layout="wide")
st.title("📊 Multi-Stock Portfolio Backtest Dashboard")

# === 1. Sidebar: Multi-file Uploads ===
st.sidebar.header("Upload Data Files")
signal_files = st.sidebar.file_uploader(
    "Upload signal_enhanced CSVs (one per stock)", type="csv", accept_multiple_files=True)
grid_files = st.sidebar.file_uploader(
    "Upload grid_search_results CSVs (one per stock)", type="csv", accept_multiple_files=True)
total_portfolio_capital = st.sidebar.number_input("Total Portfolio Capital (₹)", min_value=10000, value=100000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100

def extract_symbol(fname):
    # Adjust this logic to match your filename style. Example: 'signal_enhanced_XYZ.csv' -> 'xyz'
    return fname.split('_')[-1].split('.')[0].lower()

stock_data = {}
if signal_files and grid_files:
    signal_symbols = [extract_symbol(f.name) for f in signal_files]
    for symbol in signal_symbols:
        sig_file = next((f for f in signal_files if extract_symbol(f.name) == symbol), None)
        grid_file = next((f for f in grid_files if extract_symbol(f.name) == symbol), None)
        if sig_file and grid_file:
            df_signals = pd.read_csv(sig_file, parse_dates=['datetime'])
            df_signals.set_index('datetime', inplace=True)
            df_grid = pd.read_csv(grid_file)
            stock_data[symbol] = {'signals': df_signals, 'grid': df_grid}

symbols_list = list(stock_data.keys())
n_stocks = len(symbols_list)

# === 2. Dashboard Tabs ===
tabs = st.tabs([
    "Portfolio Overview",
    "Per Symbol Analysis",
    "All Equity Curves"
])

# === 3. Portfolio Overview Tab ===
with tabs[0]:
    if n_stocks == 0:
        st.warning("Upload matching pairs of signal_enhanced and grid_search_results files for each stock.")
    else:
        capital_per_stock = total_portfolio_capital // n_stocks
        st.write(f"Allocating ₹{capital_per_stock:,} to each of {n_stocks} stocks.")

        # -- Backtest and gather results --
        all_trades = {}
        all_equity_curves = {}
        for symbol in symbols_list:
            df_signals = stock_data[symbol]['signals']
            trades_df, equity_curve = run_backtest_simulation(
                df_signals,
                starting_capital=capital_per_stock,
                risk_per_trade=risk_per_trade
            )
            all_trades[symbol] = trades_df
            all_equity_curves[symbol] = equity_curve

        # -- Portfolio equity curve --
        portfolio_equity = None
        for symbol, equity_curve in all_equity_curves.items():
            if portfolio_equity is None:
                portfolio_equity = equity_curve
            else:
                portfolio_equity = portfolio_equity.add(equity_curve, fill_value=0)
        st.subheader("🔥 Portfolio Equity Curve")
        fig, ax = plt.subplots(figsize=(10, 5))
        if portfolio_equity is not None:
            portfolio_equity.plot(ax=ax, color="blue", linewidth=2)
            ax.set_title("Portfolio Equity Curve")
            ax.set_xlabel("Date")
            ax.set_ylabel("Total Portfolio Value (₹)")
            ax.grid(True)
            st.pyplot(fig)

        # -- Portfolio summary table --
        summary_data = []
        for symbol in symbols_list:
            trades_df = all_trades[symbol]
            final_capital = trades_df["capital_after_trade"].iloc[-1] if not trades_df.empty else capital_per_stock
            net_pnl = final_capital - capital_per_stock
            win_rate = (trades_df["net_pnl"] > 0).mean() * 100 if not trades_df.empty else 0
            summary_data.append({
                "Symbol": symbol.upper(),
                "Start Capital": capital_per_stock,
                "Final Capital": round(final_capital, 2),
                "Net PnL": round(net_pnl, 2),
                "Win Rate (%)": f"{win_rate:.2f}"
            })
        st.subheader("Portfolio Symbol Summary")
        st.dataframe(pd.DataFrame(summary_data))

# === 4. Per Symbol Analysis Tab ===
with tabs[1]:
    if n_stocks == 0:
        st.warning("Upload matching pairs of signal_enhanced and grid_search_results files for each stock.")
    else:
        symbol_select = st.selectbox(
            "Choose symbol for per-stock analysis",
            symbols_list, format_func=lambda x: x.upper())
        trades_df = all_trades[symbol_select]
        equity_curve = all_equity_curves[symbol_select]
        st.subheader(f"Trades for {symbol_select.upper()}")
        st.dataframe(trades_df)
        st.markdown("Equity Curve (per stock)")
        fig, ax = plt.subplots(figsize=(10, 4))
        equity_curve.plot(ax=ax, color="green", linewidth=2)
        ax.set_title(f"{symbol_select.upper()} Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Capital (₹)")
        ax.grid(True)
        st.pyplot(fig)

# === 5. All Equity Curves Tab ===
with tabs[2]:
    if n_stocks == 0:
        st.warning("Upload matching pairs of signal_enhanced and grid_search_results files for each stock.")
    else:
        st.subheader("All Stocks: Equity Curves")
        fig, ax = plt.subplots(figsize=(12, 6))
        for symbol, equity_curve in all_equity_curves.items():
            equity_curve.plot(ax=ax, label=symbol.upper())
        ax.set_title("Per-Stock Equity Curves")
        ax.set_xlabel("Date")
        ax.set_ylabel("Capital (₹)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
