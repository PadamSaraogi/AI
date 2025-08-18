import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from backtest import run_backtest_simulation  # Ensure this is your backtest function

st.set_page_config(layout="wide")
st.title("📊 Enhanced Multi-Stock Portfolio Backtest Dashboard")

# --- Sidebar: Multi-file Uploads and Parameters ---
st.sidebar.header("Upload Data Files")
signal_files = st.sidebar.file_uploader(
    "Upload signal_enhanced CSVs (one per stock)", type="csv", accept_multiple_files=True)
grid_files = st.sidebar.file_uploader(
    "Upload grid_search_results CSVs (one per stock)", type="csv", accept_multiple_files=True)

total_portfolio_capital = st.sidebar.number_input("Total Portfolio Capital (₹)", min_value=10000, value=100000)
risk_per_trade = st.sidebar.slider(
    "Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100

def extract_symbol(fname):
    # Adjust to your filename pattern, e.g. "signal_enhanced_ABC.csv" -> "abc"
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

tabs = st.tabs([
    "Portfolio Overview",
    "Per Symbol Analysis",
    "All Equity Curves"
])

# === Portfolio Overview Tab ===
with tabs[0]:
    if n_stocks == 0:
        st.warning("Upload matching pairs of signal_enhanced and grid_search_results files for each stock.")
    else:
        capital_per_stock = total_portfolio_capital // n_stocks
        st.write(f"Allocating ₹{capital_per_stock:,} to each of {n_stocks} stocks.")

        # Backtest each stock
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

        # Portfolio equity curve
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

        # Portfolio summary table
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
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary)

        # Highlight top performing stock
        top_symbol = df_summary.loc[df_summary['Net PnL'].idxmax(), 'Symbol']
        st.markdown(f"🏆 **Top Performing Stock:** {top_symbol}")

        # Portfolio allocation pie chart
        portfolio_capitals = df_summary["Final Capital"].tolist()
        fig2 = go.Figure(data=[go.Pie(
            labels=df_summary["Symbol"],
            values=portfolio_capitals,
            textinfo='label+percent+value',
            hole=0.2
        )])
        fig2.update_layout(title="Portfolio Allocation by Final Capital")
        st.plotly_chart(fig2)

# === Per Symbol Analysis ===
with tabs[1]:
    if n_stocks == 0:
        st.warning("Upload matching pairs of signal_enhanced and grid_search_results files for each stock.")
    else:
        symbol_select = st.selectbox(
            "Choose symbol for per-stock analysis",
            symbols_list, format_func=lambda x: x.upper())
        trades_df = all_trades[symbol_select]
        equity_curve = all_equity_curves[symbol_select]

        # Equity curve plot
        st.markdown("### Equity Curve")
        fig_eq, ax = plt.subplots(figsize=(10,4))
        equity_curve.plot(ax=ax, color="green", linewidth=2)
        ax.set_title(f"{symbol_select.upper()} Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Capital (₹)")
        ax.grid(True)
        st.pyplot(fig_eq)

        # Advanced KPIs in one row with 3 columns
        st.markdown("### Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", len(trades_df))
        col2.metric("Win Rate (%)", f"{win_rate:.2f}" if not trades_df.empty else "N/A")
        col3.metric("Net PnL", f"₹{trades_df['net_pnl'].sum():,.2f}" if not trades_df.empty else "N/A")


        st.subheader(f"Trades for {symbol_select.upper()}")

        # Date filter for trades
        min_date = trades_df['entry_time'].min()
        max_date = trades_df['exit_time'].max()
        date_range = st.date_input("Filter Trades by Date", [min_date, max_date])
        filtered_trades = trades_df[
            (trades_df['entry_time'] >= pd.to_datetime(date_range[0])) &
            (trades_df['exit_time'] <= pd.to_datetime(date_range[1]))
        ]
        st.dataframe(filtered_trades)

        # Download button for filtered trades
        csv_trade = filtered_trades.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Trades CSV",
            data=csv_trade,
            file_name=f"trades_{symbol_select}.csv",
            mime='text/csv'
        )

        # Advanced KPIs
        st.markdown("### Key Metrics")
        st.metric("Total Trades", len(trades_df))
        st.metric("Win Rate (%)", f"{win_rate:.2f}" if not trades_df.empty else "N/A")
        st.metric("Net PnL", f"₹{trades_df['net_pnl'].sum():,.2f}" if not trades_df.empty else "N/A")

        # Win/Loss pie chart
        win_counts = trades_df['net_pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss').value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=win_counts.index,
            values=win_counts.values,
            textinfo='label+percent+value',
            marker=dict(colors=['#4CAF50', '#F44336']),
            hole=0.3,
            pull=[0.1 if label=='Win' else 0 for label in win_counts.index]
        )])
        fig_pie.update_layout(title="Win/Loss Breakdown", showlegend=False)
        st.plotly_chart(fig_pie)

        # Best and Worst trades
        if not trades_df.empty:
            best_trade = trades_df.loc[trades_df['net_pnl'].idxmax()]
            worst_trade = trades_df.loc[trades_df['net_pnl'].idxmin()]
            col1, col2 = st.columns(2)
            col1.success("Best Trade")
            col1.json(best_trade.to_dict())
            col2.error("Worst Trade")
            col2.json(worst_trade.to_dict())

        # Equity curve plot
        st.markdown("### Equity Curve")
        fig_eq, ax = plt.subplots(figsize=(10,4))
        equity_curve.plot(ax=ax, color="green", linewidth=2)
        ax.set_title(f"{symbol_select.upper()} Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Capital (₹)")
        ax.grid(True)
        st.pyplot(fig_eq)

# === All Equity Curves Tab ===
with tabs[2]:
    if n_stocks == 0:
        st.warning("Upload matching pairs of signal_enhanced and grid_search_results files for each stock.")
    else:
        st.subheader("All Stocks: Equity Curves Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        for symbol, equity_curve in all_equity_curves.items():
            equity_curve.plot(ax=ax, label=symbol.upper())
        ax.set_title("Per-Stock Equity Curves")
        ax.set_xlabel("Date")
        ax.set_ylabel("Capital (₹)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
