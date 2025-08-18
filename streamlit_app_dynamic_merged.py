import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from backtest import run_backtest_simulation  # Make sure this function is accessible

st.set_page_config(layout="wide")
st.markdown(
    """
    <div style='display: flex; align-items: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/512px-React-icon.svg.png' width='60'>
        <h1 style='margin-left: 18px;'>Multi-Stock Portfolio Dashboard with Advanced Charts</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========== Sidebar: Uploads & Parameters ==========
st.sidebar.header("Upload Data Files")
signal_files = st.sidebar.file_uploader(
    "Upload signal_enhanced CSVs (one per stock)", type="csv", accept_multiple_files=True
)
grid_files = st.sidebar.file_uploader(
    "Upload grid_search_results CSVs (one per stock)", type="csv", accept_multiple_files=True
)
total_portfolio_capital = st.sidebar.number_input("Total Portfolio Capital (₹)", min_value=10000, value=100000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100


def extract_symbol(fname):
    # Adjust to your filename pattern; example: 'signal_enhanced_ABC.csv' -> 'abc'
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
    "All Equity Curves",
    "Drawdown",
    "Rolling Metrics",
    "Trade Scatter & Histogram",
    "Waterfall Chart",
    "Correlation Heatmap",
    "Trade Timeline",
    "Allocation",
])


# ========== Portfolio Overview ==========
with tabs[0]:
    if n_stocks == 0:
        st.warning("Upload matching pairs for each stock (signals + grid search results).")
    else:
        # Dynamic portfolio selection
        included_symbols = st.multiselect(
            "Select stocks included in portfolio:",
            options=symbols_list,
            default=symbols_list,
            format_func=lambda x: x.upper(),
        )
        if not included_symbols:
            st.error("Select at least one stock!")
        else:
            capital_per_stock = total_portfolio_capital // n_stocks
            st.write(f"Allocating ₹{capital_per_stock:,} to each of {n_stocks} stocks.")

            all_trades = {}
            all_equity_curves = {}
            for symbol in included_symbols:
                df_signals = stock_data[symbol]['signals']
                trades_df, equity_curve = run_backtest_simulation(
                    df_signals,
                    starting_capital=capital_per_stock,
                    risk_per_trade=risk_per_trade,
                )
                all_trades[symbol] = trades_df
                all_equity_curves[symbol] = equity_curve

            # Portfolio equity aggregation
            portfolio_equity = None
            for eq in all_equity_curves.values():
                portfolio_equity = eq if portfolio_equity is None else portfolio_equity.add(eq, fill_value=0)

            total_trades = sum([len(t) for t in all_trades.values()])
            total_net_pnl = sum([t['net_pnl'].sum() for t in all_trades.values()])

            # Portfolio drawdown & risk metrics
            if portfolio_equity is not None:
                daily_returns = portfolio_equity.pct_change().fillna(0)
                cum_returns = (1 + daily_returns).cumprod()
                drawdowns = cum_returns / cum_returns.cummax() - 1
                max_drawdown = drawdowns.min()
                sharpe = np.nan
                sortino = np.nan
                volatility = daily_returns.std()
                if volatility != 0:
                    sharpe = daily_returns.mean() / volatility * np.sqrt(252)
                    downside_std = daily_returns[daily_returns < 0].std()
                    if downside_std != 0:
                        sortino = daily_returns.mean() / downside_std * np.sqrt(252)
            else:
                max_drawdown = sharpe = sortino = volatility = np.nan

            st.markdown("### Portfolio Key Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Trades", total_trades)
            c2.metric("Net PnL (₹)", f"{total_net_pnl:,.2f}")
            c3.metric("Max Drawdown", f"{max_drawdown:.2%}" if not np.isnan(max_drawdown) else "N/A")
            c4.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
            c5.metric("Sortino Ratio", f"{sortino:.2f}" if not np.isnan(sortino) else "N/A")

            # Equity curve chart
            st.subheader("Portfolio Equity Curve")
            fig, ax = plt.subplots(figsize=(10, 5))
            portfolio_equity.plot(ax=ax, color='blue', linewidth=2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value (₹)")
            ax.grid(True)
            st.pyplot(fig)

            # Allocation pie chart
            final_values = [
                all_trades[s]["capital_after_trade"].iloc[-1] if not all_trades[s].empty else capital_per_stock
                for s in included_symbols
            ]
            fig_alloc = go.Figure(data=[go.Pie(
                labels=[s.upper() for s in included_symbols],
                values=final_values,
                hole=0.3,
                textinfo='label+percent+value'
            )])
            fig_alloc.update_layout(title="Portfolio Allocation by Final Capital")
            st.plotly_chart(fig_alloc)

            # Portfolio summary data table with download
            summary_data = []
            for symbol in included_symbols:
                trades_df = all_trades[symbol]
                final_cap = (
                    trades_df["capital_after_trade"].iloc[-1] if not trades_df.empty else capital_per_stock
                )
                net_pnl = final_cap - capital_per_stock
                win_rate = (trades_df["net_pnl"] > 0).mean() * 100 if not trades_df.empty else 0
                summary_data.append(
                    {
                        "Symbol": symbol.upper(),
                        "Start Capital": capital_per_stock,
                        "Final Capital": round(final_cap, 2),
                        "Net PnL": round(net_pnl, 2),
                        "Win Rate (%)": f"{win_rate:.2f}",
                    }
                )
            df_summary = pd.DataFrame(summary_data)
            st.subheader("Portfolio Symbol Summary")
            st.dataframe(df_summary)
            csv_summary = df_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Portfolio Summary CSV",
                csv_summary,
                "portfolio_summary.csv",
                "text/csv",
            )


# ========== Per Symbol Analysis ==========
with tabs[1]:
    if n_stocks == 0:
        st.warning("Upload data files to analyze individual stocks.")
    else:
        symbol_select = st.selectbox(
            "Select Symbol", symbols_list, format_func=lambda s: s.upper()
        )
        trades_df = all_trades.get(symbol_select)
        equity_curve = all_equity_curves.get(symbol_select)
        if trades_df is None or trades_df.empty:
            st.info("No trade data available for selected symbol.")
        else:
            win_rate = (trades_df["net_pnl"] > 0).mean() * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Trades", len(trades_df))
            c2.metric("Win Rate (%)", f"{win_rate:.2f}")
            c3.metric("Net PnL (₹)", f"{trades_df['net_pnl'].sum():,.2f}")

            # Equity curve
            st.subheader(f"{symbol_select.upper()} Equity Curve")
            fig_eq, ax = plt.subplots(figsize=(10, 4))
            equity_curve.plot(ax=ax, color="green", linewidth=2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Capital (₹)")
            ax.grid(True)
            st.pyplot(fig_eq)

            # Best/Worst trades
            best_trade = trades_df.loc[trades_df["net_pnl"].idxmax()]
            worst_trade = trades_df.loc[trades_df["net_pnl"].idxmin()]
            b_col, w_col = st.columns(2)
            b_col.success("Best Trade")
            b_col.json(best_trade.to_dict())
            w_col.error("Worst Trade")
            w_col.json(worst_trade.to_dict())

            # Filter trades by date
            min_date = trades_df["entry_time"].min()
            max_date = trades_df["exit_time"].max()
            date_range = st.date_input("Filter Trades by Date", [min_date, max_date])
            filtered_trades = trades_df[
                (trades_df["entry_time"] >= pd.to_datetime(date_range[0]))
                & (trades_df["exit_time"] <= pd.to_datetime(date_range[1]))
            ]
            st.dataframe(filtered_trades)

            csv_filtered = filtered_trades.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Filtered Trades CSV",
                csv_filtered,
                f"trades_{symbol_select}.csv",
                "text/csv",
            )

            # Win/Loss pie chart
            win_loss_counts = filtered_trades["net_pnl"].apply(lambda x: "Win" if x > 0 else "Loss").value_counts()
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=win_loss_counts.index,
                        values=win_loss_counts.values,
                        hole=0.3,
                        textinfo="label+percent+value",
                        marker=dict(colors=["#4CAF50", "#F44336"]),
                        pull=[0.1 if label == "Win" else 0 for label in win_loss_counts.index],
                    )
                ]
            )
            fig_pie.update_layout(title=f"Win/Loss Breakdown for {symbol_select.upper()}")
            st.plotly_chart(fig_pie)

            # Signal timeline scatter
            signals_df = stock_data[symbol_select]["signals"]
            if "signal" in signals_df:
                st.subheader("Signal Timeline (Buy=1 / Sell=-1 / Neutral=0)")
                color_map = {1: "green", 0: "gray", -1: "red"}
                signal_colors = signals_df["signal"].map(color_map)
                fig_sig, ax_sig = plt.subplots(figsize=(10, 2))
                ax_sig.scatter(signals_df.index, signals_df["signal"], c=signal_colors)
                ax_sig.set_yticks([-1, 0, 1])
                ax_sig.set_yticklabels(["Sell", "Neutral", "Buy"])
                ax_sig.set_title(f"Signal Timeline: {symbol_select.upper()}")
                st.pyplot(fig_sig)


# ========== All Equity Curves ==========
with tabs[2]:
    if n_stocks == 0:
        st.warning("Upload data files to analyze equity curves.")
    else:
        st.subheader("All Stocks Equity Curves")
        fig, ax = plt.subplots(figsize=(12, 6))
        for symbol, equity_curve in all_equity_curves.items():
            equity_curve.plot(ax=ax, label=symbol.upper())
        ax.set_xlabel("Date")
        ax.set_ylabel("Capital (₹)")
        ax.set_title("Per-Stock Equity Curves")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ========== Drawdown Chart ==========
with tabs[3]:
    if portfolio_equity is None:
        st.warning("Run portfolio backtest to see drawdown chart.")
    else:
        st.subheader("Portfolio Drawdown")
        cumulative_returns = portfolio_equity / portfolio_equity.iloc[0]
        drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
        fig, ax = plt.subplots(figsize=(10, 4))
        drawdowns.plot(ax=ax, color="red")
        ax.set_ylabel("Drawdown")
        ax.set_xlabel("Date")
        ax.grid(True)
        st.pyplot(fig)

# ========== Rolling Metrics ==========
with tabs[4]:
    if portfolio_equity is None:
        st.warning("Run portfolio backtest to see rolling metrics.")
    else:
        st.subheader("Rolling Performance Metrics")
        daily_ret = portfolio_equity.pct_change().fillna(0)

        rolling_window = st.slider("Rolling Window Size (Days)", min_value=5, max_value=60, value=20)
        rolling_sharpe = daily_ret.rolling(window=rolling_window).mean() / daily_ret.rolling(window=rolling_window).std() * np.sqrt(252)
        rolling_winrate = pd.Series(np.nan, index=daily_ret.index)
        # Calculate rolling win rate with a simple method
        all_trades_df = pd.concat(all_trades.values())
        # This is a simplified proxy; for exact rolling win rate, more complex logic could be implemented
        rolling_winrate.iloc[-len(rolling_sharpe):] = (
            (rolling_sharpe > 0).rolling(window=rolling_window).mean() * 100
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        rolling_sharpe.plot(ax=ax, label="Rolling Sharpe")
        ax.set_ylabel("Rolling Sharpe Ratio")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        rolling_winrate.plot(ax=ax2, label="Rolling Win Rate (%)", color="orange")
        ax2.set_ylabel("Rolling Win Rate (%)")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

# ========== Trade Scatter & Histogram ==========
with tabs[5]:
    if n_stocks == 0:
        st.warning("Upload data to view trade scatter and histogram.")
    else:
        st.subheader("Trade PnL vs Duration Scatter & PnL Histogram")
        symbol_select = st.selectbox(
            "Select Stock for Scatter Plot",
            options=symbols_list,
            format_func=lambda s: s.upper(),
        )
        trades_df = all_trades.get(symbol_select)
        if trades_df is None or trades_df.empty:
            st.info("No trade data available.")
        else:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(
                go.Scatter(
                    x=trades_df["duration_min"],
                    y=trades_df["net_pnl"],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=trades_df["net_pnl"],
                        colorscale="RdYlGn",
                        colorbar=dict(title="Net PnL"),
                        showscale=True,
                    ),
                    name="Trade PnL vs Duration",
                )
            )
            fig_scatter.update_layout(
                title=f"Trade PnL vs Duration for {symbol_select.upper()}",
                xaxis_title="Duration (minutes)",
                yaxis_title="Net PnL (₹)",
            )
            st.plotly_chart(fig_scatter)

            # PnL histogram
            fig_hist, ax = plt.subplots(figsize=(10, 4))
            ax.hist(trades_df["net_pnl"], bins=30, color="skyblue", edgecolor="black")
            ax.set_title(f"{symbol_select.upper()} Trade PnL Distribution")
            ax.set_xlabel("Net PnL")
            ax.set_ylabel("Frequency")
            st.pyplot(fig_hist)

# ========== Waterfall Chart ==========
with tabs[6]:
    st.header("Portfolio Trade-by-Trade PnL Waterfall")
    if n_stocks == 0:
        st.warning("Upload data to view waterfall chart.")
    else:
        all_trades_combined = pd.concat(all_trades.values()).sort_values("exit_time")
        if all_trades_combined.empty:
            st.info("No trades to display.")
        else:
            cum = 0
            bottoms = []
            for pnl in all_trades_combined["net_pnl"]:
                bottoms.append(cum)
                cum += pnl
            fig_water, ax = plt.subplots(figsize=(12, 5))
            ax.bar(
                range(len(all_trades_combined)),
                all_trades_combined["net_pnl"],
                bottom=bottoms,
                color=["green" if x >= 0 else "red" for x in all_trades_combined["net_pnl"]],
            )
            ax.set_xlabel("Trade Index")
            ax.set_ylabel("Net PnL (₹)")
            ax.set_title("Trade-by-Trade Net PnL Contribution")
            st.pyplot(fig_water)

# ========== Correlation Heatmap ==========
with tabs[7]:
    st.header("Portfolio Asset Correlation Heatmap")
    if n_stocks < 2:
        st.info("Upload at least two stocks to see correlation heatmap.")
    else:
        # Calculate daily returns for each stock equity curve
        returns_df = pd.DataFrame()
        for symbol, eq_curve in all_equity_curves.items():
            returns_df[symbol.upper()] = eq_curve.pct_change()
        corr = returns_df.corr()
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="Blues",
                colorbar=dict(title="Correlation"),
            )
        )
        fig_corr.update_layout(title="Correlation Matrix of Daily Returns")
        st.plotly_chart(fig_corr)

# ========== Trade Timeline ==========
with tabs[8]:
    st.header("Trade Timeline (Entry to Exit)")
    if n_stocks == 0:
        st.warning("Upload data to view trade timeline.")
    else:
        all_trades_combined = pd.concat(all_trades.values()).sort_values("entry_time")
        if all_trades_combined.empty:
            st.info("No trades to show.")
        else:
            fig_timeline, ax = plt.subplots(figsize=(12, 6))
            colors = {"Buy": "green", "Short Sell": "red"}
            for i, row in all_trades_combined.iterrows():
                ax.plot(
                    [row["entry_time"], row["exit_time"]],
                    [i, i],
                    color=colors.get(row["trade_type"], "gray"),
                    linewidth=3,
                )
            ax.set_xlabel("Date")
            ax.set_ylabel("Trade Index")
            ax.set_title("Trade Duration Timeline")
            st.pyplot(fig_timeline)

# ========== Allocation Pie Chart ==========
with tabs[9]:
    st.header("Portfolio Allocation by Final Capital")
    if n_stocks == 0:
        st.warning("Upload data to view allocation.")
    else:
        final_caps = [
            all_trades[s]["capital_after_trade"].iloc[-1] if not all_trades[s].empty else capital_per_stock
            for s in symbols_list
        ]
        fig_alloc = go.Figure(data=[go.Pie(
            labels=[s.upper() for s in symbols_list],
            values=final_caps,
            hole=0.3,
            textinfo="label+percent+value"
        )])
        fig_alloc.update_layout(title="Portfolio Capital Allocation")
        st.plotly_chart(fig_alloc)
