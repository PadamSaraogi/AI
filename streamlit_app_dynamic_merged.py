import itertools
import json
import os
import streamlit as st
import tickbus
import pandas as pd
import numpy as np
import urllib.parse
import seaborn as sns
from urllib.parse import quote_plus
from queue import Queue
import logging
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from backtest import run_backtest_simulation
import datetime
import threading
import queue
from breeze_connect import BreezeConnect
import ta
import sys
import joblib
import time
import io
import pytz  # <<< NEW
IST = pytz.timezone("Asia/Kolkata")  # <<< NEW (Gurgaon/India standard time)

st.set_page_config(layout="wide")
st.markdown("""
    <div style='display: flex; align-items: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/512px-React-icon.svg.png' width='60'>
        <h1 style='margin-left: 18px;'>Multi-Stock Trading Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

# Create two tabs: Backtesting and Live Trading
tab1, tab2 = st.tabs(["Backtesting", "Live Trading"])

with tab1:

    # Sidebar: Upload and Inputs
    st.sidebar.header("Upload Data Files")
    signal_files = st.sidebar.file_uploader(
        "Upload signal_enhanced CSVs (one per stock)", type="csv", accept_multiple_files=True)
    grid_files = st.sidebar.file_uploader(
        "Upload grid_search_results CSVs (one per stock)", type="csv", accept_multiple_files=True)
    total_portfolio_capital = st.sidebar.number_input("Total Portfolio Capital (₹)", min_value=10000, value=100000)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100

    def extract_symbol(fname):
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

    # Portfolio Overview Tab
    with tabs[0]:
        if n_stocks == 0:
            st.warning("Upload matching pairs for each stock.")
        else:
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
                all_trades = {}
                all_equity_curves = {}
                for symbol in included_symbols:
                    df_signals = stock_data[symbol]['signals']
                    trades_df, equity_curve = run_backtest_simulation(
                        df_signals,
                        starting_capital=capital_per_stock,
                        risk_per_trade=risk_per_trade
                    )
                    trades_df['symbol'] = symbol
                    all_trades[symbol] = trades_df
                    all_equity_curves[symbol] = equity_curve

                # Portfolio equity aggregation
                portfolio_equity = None
                for eq in all_equity_curves.values():
                    portfolio_equity = eq if portfolio_equity is None else portfolio_equity.add(eq, fill_value=0)

                total_trades = sum([len(t) for t in all_trades.values()])
                total_net_pnl = sum([t['net_pnl'].sum() for t in all_trades.values()])

                # Portfolio-wide metrics
                if portfolio_equity is not None:
                    daily_returns = portfolio_equity.pct_change().fillna(0)
                    cum_returns = (1 + daily_returns).cumprod()
                    drawdowns = cum_returns / cum_returns.cummax() - 1
                    max_drawdown = drawdowns.min()
                    volatility = daily_returns.std()
                    sharpe = daily_returns.mean() / volatility * np.sqrt(252) if volatility != 0 else np.nan
                    downside_std = daily_returns[daily_returns < 0].std()
                    sortino = daily_returns.mean() / downside_std * np.sqrt(252) if downside_std != 0 else np.nan
                    if portfolio_equity is not None and len(portfolio_equity) > 1:
                        start_val = portfolio_equity.iloc[0]
                        end_val = portfolio_equity.iloc[-1]
                        portfolio_return = (end_val / start_val) - 1
                    else:
                        portfolio_return = 0.0
                    portfolio_buy_hold_final_value = 0
                    buy_and_hold_start_capital = capital_per_stock * len(included_symbols) if included_symbols else 0

                    for symbol in included_symbols:
                        trades_df = all_trades[symbol]
                        signals = stock_data[symbol]["signals"].sort_index()
                        if trades_df.empty:
                            continue
                        first_time = trades_df['entry_time'].values[0]
                        last_time = trades_df['exit_time'].values[-1]
                        first_time_ts = pd.to_datetime(first_time)
                        last_time_ts = pd.to_datetime(last_time)

                        start_idx = signals.index.get_indexer([first_time_ts], method='nearest')[0]
                        end_idx = signals.index.get_indexer([last_time_ts], method='nearest')[0]

                        start_price = signals.iloc[start_idx]['close']
                        end_price = signals.iloc[end_idx]['close']

                        qty = int(capital_per_stock // start_price)
                        leftover_cash = capital_per_stock - qty * start_price

                        final_value = qty * end_price + leftover_cash
                        portfolio_buy_hold_final_value += final_value

                    if buy_and_hold_start_capital > 0:
                        buy_and_hold_return = (portfolio_buy_hold_final_value / buy_and_hold_start_capital) - 1
                    else:
                        buy_and_hold_return = 0.0
                    all_trades_concat = pd.concat(all_trades.values()) if all_trades else pd.DataFrame()

                    if not all_trades_concat.empty:
                        win_rate = (all_trades_concat['net_pnl'] > 0).mean()
                    else:
                        win_rate = 0.0
                    all_trades_concat = pd.concat(all_trades.values()) if all_trades else pd.DataFrame()

                    if not all_trades_concat.empty:
                        expectancy = all_trades_concat['net_pnl'].mean()
                    else:
                        expectancy = 0.0
                    adjusted_return = (portfolio_return - 1) * 100
                    initial_value = 0
                    final_value = 0

                    for symbol in included_symbols:
                        trades_df = all_trades[symbol]
                        if trades_df.empty:
                            continue
                        initial_value += capital_per_stock
                        final_value += trades_df['capital_after_trade'].iloc[-1]  # final symbol value

                    if initial_value > 0:
                        increase_percent = ((total_net_pnl) / total_portfolio_capital) * 100
                    else:
                        increase_percent = 0.0
                else:
                    max_drawdown = sharpe = sortino = volatility = np.nan

                st.markdown("### Portfolio Key Metrics")
                r1c1, r1c2, r1c3 = st.columns(3)
                r2c1, r2c2, r2c3 = st.columns(3)

                r1c1.metric("Total Trades", f"{total_trades}")
                r1c2.metric("Portfolio Value (₹)", f"₹{total_net_pnl + total_portfolio_capital:,.2f}")
                r1c3.metric("Returns (%)", f"{increase_percent:.2f}%")

                r2c1.metric("Buy & Hold Returns (%)", f"{buy_and_hold_return*100:.2f}%")
                r2c2.metric("Win Rate (%)", f"{win_rate*100:.2f}%")
                r2c3.metric("Expectancy (₹/Trade)", f"₹{expectancy:,.2f}")

                all_trades_combined = pd.concat(all_trades.values()).sort_values("exit_time")
                if not all_trades_combined.empty:
                    cum = 0
                    bottoms = []
                    for pnl in all_trades_combined["net_pnl"]:
                        bottoms.append(cum)
                        cum += pnl

                    qty_map = {}
                    for symbol in included_symbols:
                        trades_df = all_trades[symbol]
                        signals = stock_data[symbol]["signals"].sort_index()
                        if trades_df.empty:
                            qty_map[symbol] = 0
                            continue
                        first_time = trades_df['entry_time'].values[0]
                        first_time_ts = pd.to_datetime(first_time)
                        start_idx = signals.index.get_indexer([first_time_ts], method='nearest')[0]
                        start_price = signals.iloc[start_idx]['close']
                        qty_map[symbol] = int(capital_per_stock // start_price)

                    buy_hold_pnl_over_time = []
                    initial_portfolio_value = capital_per_stock * len(included_symbols)

                    for _, row in all_trades_combined.iterrows():
                        timestamp = row['exit_time']
                        ts = pd.to_datetime(timestamp)
                        portfolio_value = 0
                        for symbol in included_symbols:
                            signals = stock_data[symbol]["signals"].sort_index()
                            if ts < signals.index[0]:
                                current_price = signals.iloc[0]['close']
                            elif ts > signals.index[-1]:
                                current_price = signals.iloc[-1]['close']
                            else:
                                pos = signals.index.get_indexer([ts], method='ffill')[0]
                                current_price = signals.iloc[pos]['close']
                            qty = qty_map[symbol]
                            portfolio_value += qty * current_price
                        buy_hold_pnl = portfolio_value - initial_portfolio_value
                        buy_hold_pnl_over_time.append(buy_hold_pnl)

                    fig_water, ax_water = plt.subplots(figsize=(12, 4))
                    ax_water.bar(
                        range(len(all_trades_combined)),
                        all_trades_combined["net_pnl"],
                        bottom=bottoms,
                        color=["green" if x >= 0 else "red" for x in all_trades_combined["net_pnl"]],
                        label='Strategy Cumulative'
                    )
                    ax_water.plot(
                        range(len(all_trades_combined)),
                        buy_hold_pnl_over_time,
                        color='blue',
                        linewidth=2,
                        label='Buy & Hold'
                    )
                    ax_water.set_xlabel("Trade Index")
                    ax_water.set_ylabel("Net PnL (₹)")
                    ax_water.set_title("Trade-by-Trade Net PnL Contribution (Portfolio) with Buy & Hold Over Time")
                    ax_water.legend()
                    st.pyplot(fig_water)

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

                st.subheader("Portfolio Drawdown")
                fig_dd, ax_dd = plt.subplots(figsize=(10, 4))
                drawdowns.plot(ax=ax_dd, color="red")
                ax_dd.set_ylabel("Drawdown")
                ax_dd.set_xlabel("Date")
                ax_dd.grid(True)
                st.pyplot(fig_dd)

                # Portfolio leaderboard
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
                            "Win Rate (%)": f"{win_rate:.2f}"
                        }
                    )
                df_summary = pd.DataFrame(summary_data)

                # Collect and align daily returns for all stocks
                returns_df = pd.DataFrame()
                for symbol, eq_curve in all_equity_curves.items():
                    returns_df[symbol.upper()] = eq_curve.pct_change()

                returns_corr = returns_df.corr()

                st.markdown("### Correlation Heatmap of Daily Returns")

                # Ensure all values are finite and matrix isn't empty
                if not returns_corr.empty and np.isfinite(returns_corr.values).all():
                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                    sns.heatmap(returns_corr, annot=True, cmap="RdBu", center=0, linewidths=.5, fmt=".2f", ax=ax_corr)
                    ax_corr.set_title("Correlation Heatmap (Daily Returns)")
                    st.pyplot(fig_corr)
                else:
                    st.info("Not enough data to display correlation heatmap. Please upload several stocks with sufficient history.")

                window = 21  # About a month
                risk_free = 0  # Change if you'd like

                if portfolio_equity is not None:
                    rets = portfolio_equity.pct_change().dropna()
                    rolling_sharpe = rets.rolling(window).mean() / rets.rolling(window).std() * np.sqrt(252)
                    rolling_downside = rets.where(rets < 0, 0)
                    rolling_sortino = rets.rolling(window).mean() / rolling_downside.rolling(window).std() * np.sqrt(252)

                    st.markdown("### Rolling Sharpe & Sortino Ratios (Portfolio)")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    rolling_sharpe.plot(ax=ax, label='Sharpe Ratio')
                    rolling_sortino.plot(ax=ax, label='Sortino Ratio')
                    ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
                    ax.set_ylabel("Ratio (annualized)")
                    ax.set_xlabel("Date")
                    ax.legend()
                    ax.set_title(f"Rolling {window}-Day Portfolio Sharpe/Sortino")
                    ax.grid(True)
                    st.pyplot(fig)
                else:
                    st.info("Portfolio equity curve not available for rolling ratios.")

                import plotly.express as px

                st.markdown("### Top Contributors to Portfolio PnL - Interactive Chart & Highlights")

                # Prepare data as before
                contrib_list = []
                total_pnl = sum(
                    td['net_pnl'].sum() for td in all_trades.values()
                    if not td.empty and 'net_pnl' in td.columns
                )

                for symbol, trades_df in all_trades.items():
                    net_pnl = trades_df['net_pnl'].sum() if not trades_df.empty and 'net_pnl' in trades_df.columns else 0
                    contrib_pct = (net_pnl / total_pnl * 100) if total_pnl != 0 else 0
                    num_trades = len(trades_df) if not trades_df.empty else 0
                    avg_pnl = net_pnl / num_trades if num_trades > 0 else 0
                    contrib_list.append({
                        'Symbol': symbol.upper(),
                        'Net PnL': net_pnl,
                        'Contribution (%)': contrib_pct,
                        'Trades': num_trades,
                        'Avg Trade PnL': avg_pnl
                    })

                df_contrib = pd.DataFrame(contrib_list)
                df_contrib.sort_values('Contribution (%)', ascending=True, inplace=True)  # Ascending for horizontal bar

                # Assign colors based on Net PnL sign
                df_contrib['Color'] = df_contrib['Net PnL'].apply(lambda x: 'green' if x >= 0 else 'red')

                # 1. Interactive Horizontal Bar Chart
                fig = px.bar(
                    df_contrib,
                    x='Contribution (%)',
                    y='Symbol',
                    orientation='h',
                    text=df_contrib['Contribution (%)'].map('{:.2f}%'.format),
                    color='Color',
                    color_discrete_map={'green': 'green', 'red': 'red'},
                    hover_data={
                        'Net PnL': ':.2f',
                        'Trades': True,
                        'Avg Trade PnL': ':.2f',
                        'Color': False
                    }
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    showlegend=False,
                    xaxis_title='Contribution (%)',
                    yaxis_title='Stock Symbol',
                    margin=dict(l=0, r=20, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

                # 2. Highlight Top 5 Contributors with Cards & Progress Bars
                st.markdown("### Top 5 Contributors Quick Stats")
                top_n = min(5, len(df_contrib))
                df_top = df_contrib.sort_values('Contribution (%)', ascending=False).head(top_n).reset_index(drop=True)

                for i in range(top_n):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.metric(label="Symbol", value=df_top.iloc[i]['Symbol'])
                    with col2:
                        contribution = df_top.iloc[i]['Contribution (%)']
                        st.write(f"Contribution: **{contribution:.2f}%**")
                        percent = max(min(contribution, 100), 0)  # Clamp between 0 and 100
                        st.progress(percent / 100)
                    st.write(
                        f"Net PnL: ₹{df_top.iloc[i]['Net PnL']:.2f} | Trades: {df_top.iloc[i]['Trades']} | "
                        f"Avg Trade PnL: ₹{df_top.iloc[i]['Avg Trade PnL']:.2f}"
                    )

    # Per Symbol Analysis Tab
    with tabs[1]:
        if n_stocks == 0:
            st.warning("Upload data files to analyze individual stocks.")
        else:
            symbol_select = st.selectbox(
                "Select Symbol", symbols_list, format_func=lambda s: s.upper()
            )
            capital_per_stock = total_portfolio_capital // n_stocks
            trades_df, equity_curve = run_backtest_simulation(
                stock_data[symbol_select]['signals'],
                starting_capital=capital_per_stock,
                risk_per_trade=risk_per_trade,
            )
            st.write(f"Number of trades: {len(trades_df)}")   # Debug line

            win_rate = (trades_df["net_pnl"] > 0).mean() * 100 if not trades_df.empty else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Trades", len(trades_df))
            c2.metric("Win Rate (%)", f"{win_rate:.2f}")
            c3.metric("Net PnL (₹)", f"{trades_df['net_pnl'].sum():,.2f}" if not trades_df.empty else "0.00")

            st.subheader(f"{symbol_select.upper()} Equity Curve")
            fig_eq, ax = plt.subplots(figsize=(10, 4))
            equity_curve.plot(ax=ax, color="green", linewidth=2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Capital (₹)")
            ax.grid(True)
            st.pyplot(fig_eq)

            st.subheader(f"Candlestick Chart with Trades ({symbol_select.upper()})")
            signals_df = stock_data[symbol_select]['signals']

            if {'open', 'high', 'low', 'close'}.issubset(signals_df.columns):
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=signals_df.index,
                    open=signals_df['open'],
                    high=signals_df['high'],
                    low=signals_df['low'],
                    close=signals_df['close']
                )])

                fig_candle.add_trace(go.Scatter(
                    x=trades_df['entry_time'],
                    y=trades_df['entry_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=9),
                    name='Buy Entry'
                ))

                fig_candle.add_trace(go.Scatter(
                    x=trades_df['exit_time'],
                    y=trades_df['final_exit_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=9),
                    name='Exit'
                ))

                fig_candle.update_layout(
                    title=f"{symbol_select.upper()} Price & Trades",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    autosize=True,
                    margin=dict(l=0, r=0, t=30, b=0)
                )

                st.plotly_chart(fig_candle, use_container_width=True)

            st.subheader(f"{symbol_select.upper()} Drawdown")
            eq_cumret = equity_curve / equity_curve.iloc[0]
            drawdowns = eq_cumret / eq_cumret.cummax() - 1
            fig_dd, ax_dd = plt.subplots(figsize=(10, 4))
            drawdowns.plot(ax=ax_dd, color="red")
            ax_dd.set_ylabel("Drawdown")
            ax_dd.set_xlabel("Date")
            ax_dd.grid(True)
            st.pyplot(fig_dd)

            # Convert to datetime if needed (keep timezone-naive here since source is historical)
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

            # Filter for intraday trades only (start and end on same day)
            intraday_trades_df = trades_df[
                trades_df['entry_time'].dt.date == trades_df['exit_time'].dt.date
            ].copy()

            st.write(f"Number of intraday trades: {len(intraday_trades_df)}")

            if not intraday_trades_df.empty:
                st.subheader(f"Intraday Trades for {symbol_select.upper()}")
                st.dataframe(intraday_trades_df.sort_values('exit_time').reset_index(drop=True))

                # Optional CSV download
                csv_download = intraday_trades_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    f"Download {symbol_select.upper()} Intraday Trades as CSV",
                    csv_download,
                    file_name=f"{symbol_select}_intraday_trades.csv",
                    mime="text/csv"
                )
            else:
                st.info("No intraday trade data available for selected symbol.")

    # All Equity Curves Tab
    with tabs[2]:
        if n_stocks == 0:
            st.warning("Upload data files to compare equity curves.")
        else:
            capital_per_stock = total_portfolio_capital // n_stocks
            all_trades = {}
            all_equity_curves = {}

            # Run backtest per stock and collect trades & equity curves
            for symbol in symbols_list:
                trades_df, eq_curve = run_backtest_simulation(
                    stock_data[symbol]['signals'],
                    starting_capital=capital_per_stock,
                    risk_per_trade=risk_per_trade,
                )
                all_trades[symbol] = trades_df
                all_equity_curves[symbol] = eq_curve

            # --- 5. New: Interactive Normalized Equity Curves Including Portfolio (Plotly) ---
            fig5 = go.Figure()

            for symbol, eq_curve in all_equity_curves.items():
                eq_norm = eq_curve / eq_curve.iloc[0] * 100
                fig5.add_trace(go.Scatter(
                    x=eq_norm.index,
                    y=eq_norm.values,
                    mode="lines",
                    name=symbol.upper(),
                    line=dict(width=2),
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}"
                ))

            # portfolio_equity is defined in earlier tab; safe guard if present
            if 'portfolio_equity' in locals() and portfolio_equity is not None and len(portfolio_equity) > 1:
                portfolio_equity_norm = portfolio_equity / portfolio_equity.iloc[0] * 100
                fig5.add_trace(go.Scatter(
                    x=portfolio_equity_norm.index,
                    y=portfolio_equity_norm.values,
                    mode="lines",
                    name="PORTFOLIO",
                    line=dict(width=2, color='white'),
                    hovertemplate="Portfolio<br>%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}"
                ))

            fig5.update_layout(
                title="Normalized Equity Curves (Including Portfolio)",
                xaxis_title="Date",
                yaxis_title="Normalized Capital (Start = 100)",
                hovermode="x unified",
                legend_title="Legend",
                height=600,
                template="plotly_white",
            )
            st.plotly_chart(fig5, use_container_width=True)

            perf_summary = []
            for symbol, eq_curve in all_equity_curves.items():
                total_return_pct = (eq_curve.iloc[-1] / eq_curve.iloc[0] - 1) * 100 if len(eq_curve) > 1 else 0
                drawdown_pct = ((eq_curve / eq_curve.cummax()) - 1).min() * 100 if len(eq_curve) > 1 else 0
                daily_rets = eq_curve.pct_change().dropna()
                sharpe_ratio = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else np.nan

                days = (eq_curve.index[-1] - eq_curve.index[0]).days
                cagr = ((eq_curve.iloc[-1] / eq_curve.iloc[0]) ** (365 / days) - 1) * 100 if days > 0 else 0
                calmar = cagr / abs(drawdown_pct) if drawdown_pct != 0 else np.nan

                perf_summary.append({
                    "Symbol": symbol.upper(),
                    "CAGR (%)": cagr,
                    "Total Return (%)": total_return_pct,
                    "Max Drawdown (%)": drawdown_pct,
                    "Calmar Ratio": calmar,
                    "Sharpe Ratio": sharpe_ratio,
                })

            df_perf = pd.DataFrame(perf_summary)
            for col in ["CAGR (%)", "Total Return (%)", "Max Drawdown (%)", "Calmar Ratio", "Sharpe Ratio"]:
                df_perf[col] = df_perf[col].astype(float).map("{:.2f}".format)

            st.markdown("### Advanced Performance Summary")
            st.dataframe(df_perf)

            def calculate_streaks(profits):
                streaks = []
                cur_streak = 0
                prev_win = None
                for pnl in profits:
                    win = pnl > 0
                    if win == prev_win:
                        cur_streak += 1
                    else:
                        if prev_win is not None:
                            streaks.append((prev_win, cur_streak))
                        cur_streak = 1
                        prev_win = win
                streaks.append((prev_win, cur_streak))
                return streaks

            if all_trades:
                all_trades_concat = pd.concat(all_trades.values())
                profits = all_trades_concat['net_pnl'] > 0
                streaks = calculate_streaks(all_trades_concat['net_pnl'].values)

                st.markdown("### Win/Loss Streaks")
                wins = [length for win, length in streaks if win]
                losses = [length for win, length in streaks if not win]

                fig, ax = plt.subplots(figsize=(12, 4))
                ax.hist(wins, bins=range(1, max(wins)+2) if wins else [1,2], alpha=0.7, label='Winning Streaks', color='green')
                ax.hist(losses, bins=range(1, max(losses)+2) if losses else [1,2], alpha=0.7, label='Losing Streaks', color='red')
                ax.set_xlabel('Streak Length (Number of Trades)')
                ax.set_ylabel('Frequency')
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("No trade data for streak analysis.")

            st.markdown("### Outlier Trades - Top Winning & Losing Intraday Trades")

            all_trades_combined = []
            for symbol, trades_df in all_trades.items():
                if trades_df.empty:
                    continue
                temp_df = trades_df.copy()
                temp_df['Symbol'] = symbol.upper()
                all_trades_combined.append(temp_df)

            if all_trades_combined:
                combined_df = pd.concat(all_trades_combined)

                combined_df['entry_time'] = pd.to_datetime(combined_df['entry_time'])
                combined_df['exit_time'] = pd.to_datetime(combined_df['exit_time'])
                combined_df = combined_df[combined_df['entry_time'].dt.date == combined_df['exit_time'].dt.date]

                combined_df['entry_price_safe'] = combined_df['entry_price'] if 'entry_price' in combined_df.columns else pd.NA

                exit_price = combined_df['exit_price'] if 'exit_price' in combined_df.columns else None
                final_exit_price = combined_df['final_exit_price'] if 'final_exit_price' in combined_df.columns else None

                if exit_price is not None and final_exit_price is not None:
                    combined_df['exit_price_safe'] = exit_price.fillna(final_exit_price)
                elif exit_price is not None:
                    combined_df['exit_price_safe'] = exit_price
                elif final_exit_price is not None:
                    combined_df['exit_price_safe'] = final_exit_price
                else:
                    combined_df['exit_price_safe'] = pd.NA

                combined_df['entry_time_fmt'] = combined_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
                combined_df['exit_time_fmt'] = combined_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M')

                top_winning = combined_df.nlargest(5, 'net_pnl').reset_index(drop=True)
                top_losing = combined_df.nsmallest(5, 'net_pnl').reset_index(drop=True)

                css = """
                <style>
                .card-container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 1rem;
                    padding: 1rem 0;
                    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                }
                .card {
                    flex: 1 1 280px;
                    max-width: 320px;
                    min-height: 230px;
                    border-radius: 12px;
                    padding: 20px;
                    box-sizing: border-box;
                    color: white;
                    text-align: center;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    box-shadow: 0 0 5px #39ff14, 0 0 10px #39ff14, 0 0 20px #39ff14;
                    transition: box-shadow 0.3s ease;
                    background-color: #39ff14;
                    margin-bottom: 1rem;
                }
                .card.loser {
                    box-shadow: 0 0 5px #ff073a, 0 0 10px #ff073a, 0 0 20px #ff073a;
                    background-color: #ff073a;
                }
                .card:hover {
                    box-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 40px #00ff00;
                }
                .card.loser:hover {
                    box-shadow: 0 0 10px #ff1744, 0 0 20px #ff1744, 0 0 40px #ff1744;
                }

                .card h4 {
                    margin-bottom: 15px;
                    font-weight: 700;
                    font-size: 1.25rem;
                    text-shadow: 0 0 7px rgba(0,0,0,0.7);
                }
                .card p {
                    margin: 5px 0;
                    font-size: 0.95rem;
                    font-weight: 600;
                    text-shadow: 0 0 5px rgba(0,0,0,0.6);
                }
                </style>
                """

                def make_card_html(trade, is_winner=True):
                    card_class = "card" if is_winner else "card loser"
                    emoji = "🏆" if is_winner else "⚠️"
                    ep = trade['entry_price_safe']
                    xp = trade['exit_price_safe']
                    ep_str = f"₹{ep:,.2f}" if pd.notna(ep) else "N/A"
                    xp_str = f"₹{xp:,.2f}" if pd.notna(xp) else "N/A"

                    return f"""
                    <div class="{card_class}">
                        <h4>{emoji} {trade['Symbol']} - ₹{trade['net_pnl']:,.2f} {'Profit' if is_winner else 'Loss'}</h4>
                        <p><strong>Entry Time:</strong> {trade['entry_time_fmt']}</p>
                        <p><strong>Exit Time:</strong> {trade['exit_time_fmt']}</p>
                        <p><strong>Entry Price:</strong> {ep_str} | <strong>Exit Price:</strong> {xp_str}</p>
                        <p><strong>Trade PnL:</strong> ₹{trade['net_pnl']:,.2f}</p>
                    </div>
                    """

                def render_cards(title, df, is_winner):
                    cards_html = "".join(make_card_html(df.iloc[i], is_winner) for i in range(len(df)))
                    full_html = f"""
                    {css}
                    <h4>{title}</h4>
                    <div class='card-container'>
                        {cards_html}
                    </div>
                    """
                    components.html(full_html, height=700)

                render_cards("Top 5 Winning Intraday Trades", top_winning, True)
                render_cards("Top 5 Losing Intraday Trades", top_losing, False)

            else:
                st.info("No intraday trades data available to display outlier trades.")

with tab2:


# ================= Hard-coded Breeze credentials (fill these in code) =================
# ⚠️ Put your real keys here (do NOT commit them to git)
    BREEZE_API_KEY    = "=4c730660p24@d03%65343MG909o217L"
    BREEZE_API_SECRET = "416D2gJdy064P7F7)s5e590J8I1692~7"

def _keys_ok() -> bool:
    bad = (
        not BREEZE_API_KEY or BREEZE_API_KEY.startswith("PUT_") or
        not BREEZE_API_SECRET or BREEZE_API_SECRET.startswith("PUT_")
    )
    if bad:
        st.error("Set BREEZE_API_KEY and BREEZE_API_SECRET in the code at the top of this block.")
    return not bad

# ================= UI Title =================
st.title("📊 Live Trading Dashboard")

# ================= Constants & logging =================
MAX_WINDOW_SIZE  = 15000
RENDER_SLEEP_SEC = 1
IDLE_REFRESH_SEC = 3.0

lg = logging.getLogger("LiveTradingLogger")
lg.setLevel(logging.INFO)
if not lg.handlers:
    fh = logging.FileHandler("live_trading.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    lg.addHandler(fh)

# Ensure BUS_ID even if an older tickbus is imported by accident
if not hasattr(tickbus, "BUS_ID"):
    import os, uuid as _uuid
    tickbus.BUS_ID = os.environ.get("TICKBUS_ID", str(_uuid.uuid4())[:8])

lg.info(f"[boot bus {tickbus.BUS_ID}] tab loaded")

# ✅ start 1-Hz (configurable) bar aggregator once
try:
    tickbus.enable_debug(False)                 # set True if you want stdout bar logs
    tickbus.start_bar_aggregator(cadence_sec=1) # 1-second bars
except Exception as _e:
    lg.info(f"tickbus aggregator start: {_e}")

# ================= Session defaults =================
defaults = {
    "live_data": pd.DataFrame(),   # 1 row per emitted 1s bar
    "position": None,              # {"side": "long"/"short", "entry_price": float, "entry_time": ts}
    "trades": [],                  # list of closed trades dicts
    "equity_curve": [],            # [{timestamp, total_pnl}]
    "model": None,
    "breeze": None,
    "last_bars": [],               # preview last few bars (json-safe)
    "run_live": False,
    "last_render_ts": 0.0,
    # decisions
    "auto_trade": True,
    "conf_threshold": 0.55,
    "adx_target_mult": 0.0,
    "trail_mult": 0.0,
    "time_limit": 0,
    "last_action_signal": None,
    "last_decision": None,
    "decision_history": [],
    "allow_shorts": True,          # enable short selling
    # connection fields
    "exchange_code": "",
    "stock_code": "",
    "stock_token": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= Utilities =================
IST = pytz.timezone("Asia/Kolkata")

# Safe import for `ta` (indicators)
try:
    import ta
except Exception:  # keep the app running even if ta missing
    class _TAStub: pass
    ta = _TAStub()
    setattr(ta, "trend", _TAStub())
    setattr(ta, "volatility", _TAStub())
    setattr(ta, "momentum", _TAStub())

def _to_jsonable(obj):
    try:
        json.dumps(obj); return obj
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    try:
        import numpy as _np
        if isinstance(obj, _np.generic): return obj.item()
    except Exception:
        pass
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp): return obj.isoformat()
    except Exception:
        pass
    return str(obj)

# ================= Indicators / features =================
def calculate_indicators_live(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy().sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    if getattr(d["timestamp"].dt, "tz", None) is None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(IST)
    d["last_traded_price"] = pd.to_numeric(d["last_traded_price"], errors="coerce").ffill()
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)

    # quick EMAs/RSI/ATR on bar close (guard if ta missing)
    price = d["last_traded_price"]
    try:
        if hasattr(ta, "trend") and hasattr(ta.trend, "ema_indicator"):
            if len(d) >= 20: d["ema_20"] = ta.trend.ema_indicator(price, window=20)
            if len(d) >= 50: d["ema_50"] = ta.trend.ema_indicator(price, window=50)
        if hasattr(ta, "volatility") and hasattr(ta.volatility, "average_true_range") and len(d) >= 14:
            d["ATR"] = ta.volatility.average_true_range(high=price, low=price, close=price, window=14)
        if hasattr(ta, "momentum") and hasattr(ta.momentum, "rsi") and len(d) >= 14:
            d["RSI"] = ta.momentum.rsi(price, window=14)
    except Exception as _e:
        lg.info(f"indicator note: {_e}")

    # lightweight extras
    d["hour_of_day"] = d["timestamp"].dt.hour.astype("float64")
    v = pd.to_numeric(d["volume"], errors="coerce")
    d["volume_spike_ratio"] = (v / v.rolling(200, min_periods=20).mean()).astype("float64")

    # Optional: 1-minute resample features
    try:
        bars_1m = (
            d.set_index("timestamp")
             .resample("1min")
             .agg(open=("last_traded_price","first"),
                  high=("last_traded_price","max"),
                  low =("last_traded_price","min"),
                  close=("last_traded_price","last"),
                  vol  =("volume","sum"))
             .dropna(subset=["open","high","low","close"])
        )
        if not bars_1m.empty and hasattr(ta, "trend") and hasattr(ta, "volatility"):
            adx = ta.trend.adx(bars_1m["high"], bars_1m["low"], bars_1m["close"], window=14)
            bb  = ta.volatility.BollingerBands(bars_1m["close"], window=20, window_dev=2)
            bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            ret_1h = (bars_1m["close"] / bars_1m["close"].shift(60) - 1.0)
            feat = pd.DataFrame({"ADX14": adx, "bb_width": bb_width, "return_1h": ret_1h}).dropna(how="all")
            d = pd.merge_asof(
                d.sort_values("timestamp"),
                feat.sort_index(),
                left_on="timestamp",
                right_index=True,
                direction="backward",
                allow_exact_matches=True,
            )
    except Exception as _e:
        lg.info(f"feature block note: {_e}")
    return d

# ================= Model prediction =================
def predict_signal(model, df: pd.DataFrame):
    if model is None or df.empty: return None, None
    req = list(getattr(model, "feature_names_in_", [])) or \
          ['ema_20','ema_50','ATR','RSI','ADX14','bb_width','hour_of_day','return_1h','volume_spike_ratio']
    if any(c not in df.columns for c in req): return None, None
    latest = df.dropna(subset=[c for c in req if c != "volume_spike_ratio"])
    if latest.empty: return None, None
    X = latest.iloc[-1:][req]
    pred = model.predict(X)[0]
    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes_ = getattr(model, "classes_", None)
        buy_idx = None
        if classes_ is not None:
            try:
                buy_idx = list(classes_).index(1)
            except Exception:
                for i, c in enumerate(classes_):
                    if str(c).lower() in ("1","buy","long","open"): buy_idx = i; break
        conf = float(proba[buy_idx]) if buy_idx is not None else float(proba.max())
    return pred, conf

def _norm_signal(pred) -> int:
    if pred is None: return 0
    try:
        val = int(pred)
        if val in (-1,0,1): return val
        if val == 2: return -1
        if val == 1: return 1
        return 0
    except Exception:
        s = str(pred).strip().lower()
        if s in ("buy","long","open","enter_long","go_long"): return 1
        if s in ("sell","short","close","exit","go_short"): return -1
        return 0

# ================= Trades state (two-sided) =================
def update_trades(signal, price, timestamp):
    """
    signal: +1 = go long / close short, -1 = go short / close long
    """
    pos = st.session_state.position
    side = pos["side"] if pos else None

    def _pnl_for(side_, entry, px):
        return (px - entry) if side_ == "long" else (entry - px)

    if pos is None:
        if signal == 1:
            st.session_state.position = {"side": "long", "entry_price": float(price), "entry_time": timestamp}
            logging.getLogger("LiveTradingLogger").info(f"Open LONG @ {price} on {timestamp}")
        elif signal == -1 and st.session_state.allow_shorts:
            st.session_state.position = {"side": "short", "entry_price": float(price), "entry_time": timestamp}
            logging.getLogger("LiveTradingLogger").info(f"Open SHORT @ {price} on {timestamp}")
        # mark-to-market after any open attempt
        total = sum(t['pnl'] for t in st.session_state.trades)
        if st.session_state.position is not None:
            total += _pnl_for(st.session_state.position["side"], float(st.session_state.position["entry_price"]), float(price))
        st.session_state.equity_curve.append({"timestamp": timestamp, "total_pnl": float(total)})
        return

    if signal == 1 and side == "short":
        pnl = _pnl_for("short", float(pos["entry_price"]), float(price))
        st.session_state.trades.append({
            "side": "short", "entry_price": float(pos["entry_price"]), "exit_price": float(price),
            "entry_time": pos["entry_time"], "exit_time": timestamp, "pnl": float(pnl),
        })
        st.session_state.position = None
        logging.getLogger("LiveTradingLogger").info(f"Close SHORT @ {price} on {timestamp} | PnL {pnl:.2f}")
    elif signal == -1 and side == "long":
        pnl = _pnl_for("long", float(pos["entry_price"]), float(price))
        st.session_state.trades.append({
            "side": "long", "entry_price": float(pos["entry_price"]), "exit_price": float(price),
            "entry_time": pos["entry_time"], "exit_time": timestamp, "pnl": float(pnl),
        })
        st.session_state.position = None
        logging.getLogger("LiveTradingLogger").info(f"Close LONG @ {price} on {timestamp} | PnL {pnl:.2f}")

    # mark-to-market equity curve
    total = sum(t['pnl'] for t in st.session_state.trades)
    if st.session_state.position is not None:
        total += _pnl_for(st.session_state.position["side"], float(st.session_state.position["entry_price"]), float(price))
    st.session_state.equity_curve.append({"timestamp": timestamp, "total_pnl": float(total)})

# ================= Broker tick callback → aggregator =================
def on_ticks(*args, **kwargs):
    """Broker websocket callback → push raw ticks into tickbus aggregator."""
    ticks = kwargs.get("ticks") if "ticks" in kwargs else (args[0] if args else (kwargs if kwargs else None))
    batch = ticks if isinstance(ticks, list) else [ticks]
    fed = 0
    for item in batch:
        if item is None or not isinstance(item, dict):
            continue
        # extract price/size/time best-effort
        ts = item.get("ltt") or item.get("last_trade_time") or item.get("exchange_time") \
             or item.get("trade_time") or item.get("time") or item.get("timestamp") \
             or item.get("datetime") or item.get("created_at")
        try:
            ts = pd.to_datetime(ts, utc=True, errors="coerce")
            ts = ts.timestamp() if pd.notna(ts) else time.time()
        except Exception:
            ts = time.time()
        px = None
        for k in ["last","Last","LAST","last_traded_price","LastTradedPrice","lastTradedPrice",
                  "ltp","LTP","lastPrice","LastPrice","close","Close","price","Price"]:
            if k in item and item[k] not in (None, ""):
                px = item[k]; break
        if px is None:
            continue
        try:
            px = float(str(px).replace(",", ""))
        except Exception:
            continue
        sz = item.get("volume") or item.get("qty") or 1.0
        try:
            sz = float(str(sz).replace(",", ""))
        except Exception:
            sz = 1.0

        tickbus.put_raw_tick({"ts": ts, "price": px, "size": sz})
        fed += 1

    logging.getLogger("LiveTradingLogger").info(f"[bus {tickbus.BUS_ID}] raw → aggregator ({fed} items)")

# Rebind on rerun if already connected
if st.session_state.get("breeze") is not None:
    try:
        st.session_state.breeze.on_ticks = on_ticks
        ec = st.session_state.get("exchange_code", "")
        sc = st.session_state.get("stock_code", "")
        stkn = st.session_state.get("stock_token", "")
        if stkn:
            st.session_state.breeze.subscribe_feeds(stock_token=stkn.strip(), get_market_depth=True, get_exchange_quotes=True)
        elif ec and sc:
            st.session_state.breeze.subscribe_feeds(exchange_code=ec, stock_code=sc.strip(), product_type="cash",
                                                    get_market_depth=True, get_exchange_quotes=True)
    except Exception as _e:
        lg.info(f"Rebind/resubscribe note: {_e}")

# ================= Bar processor (UI thread) =================
def process_bar_queue():
    """
    Drain 1-second bars from tickbus and update state.
    Returns number of bars processed.
    """
    processed = 0
    bars = tickbus.drain_bars(max_items=10000)
    if not bars:
        return 0

    # append to live_data
    rows = []
    for b in bars:
        # use bar end time as timestamp for decisions/plots
        ts = pd.to_datetime(b["end_ts"], unit="s", utc=True).tz_convert(IST)
        rows.append({
            "timestamp": ts,
            "last_traded_price": float(b["close"]),
            "volume": float(b.get("volume", 0.0)),
            "raw": json.dumps(b),   # keep bar payload for debugging
        })
    df_new = pd.DataFrame(rows)
    st.session_state.live_data = (
        pd.concat([st.session_state.live_data, df_new], ignore_index=True)
        .tail(MAX_WINDOW_SIZE)
        .reset_index(drop=True)
    )
    processed = len(df_new)

    if processed > 0:
        df = calculate_indicators_live(st.session_state.live_data.copy())
        st.session_state.live_data = df

        # ===== Decision layer (two-sided; one decision per bar) =====
        if st.session_state.model is not None and not df.empty:
            pred, conf = predict_signal(st.session_state.model, df)
            sig = _norm_signal(pred)

            def adx_gate_ok(dff):
                if "ADX14" not in dff.columns or st.session_state.adx_target_mult <= 0: return True
                s = pd.to_numeric(dff["ADX14"], errors="coerce").dropna()
                if len(s) < 20: return True
                return float(s.iloc[-1]) >= st.session_state.adx_target_mult * float(s.tail(200).median())

            def time_limit_exit(now_ts_):
                tl = int(st.session_state.time_limit or 0)
                if tl <= 0 or st.session_state.position is None: return False
                held = (now_ts_ - pd.to_datetime(st.session_state.position["entry_time"])).total_seconds() / 60
                return held >= tl

            now_ts = pd.to_datetime(df["timestamp"].iloc[-1])
            last_px = float(pd.to_numeric(df["last_traded_price"].iloc[-1], errors="coerce"))
            pos = st.session_state.position
            pos_open = pos is not None
            pos_side = pos["side"] if pos_open else None

            final_signal, reason = 0, "none"

            # 1) time limit exits ANY side
            if time_limit_exit(now_ts):
                final_signal = 1 if (pos_open and pos_side == "short") else (-1 if pos_open else 0)
                reason = "time_limit"

            # 2) model signal (symmetric)
            elif sig != 0 and conf is not None and conf >= float(st.session_state.conf_threshold) and adx_gate_ok(df):
                if sig == 1:
                    if not pos_open: final_signal, reason = 1, "model_open_long"
                    elif pos_side == "short": final_signal, reason = 1, "model_close_short"
                elif sig == -1:
                    if not pos_open and st.session_state.allow_shorts:
                        final_signal, reason = -1, "model_open_short"
                    elif pos_side == "long":
                        final_signal, reason = -1, "model_close_long"

            # 3) EMA fallback (symmetric)
            elif {"ema_20","ema_50"}.issubset(df.columns):
                e20, e50 = df["ema_20"].iloc[-1], df["ema_50"].iloc[-1]
                if pd.notna(e20) and pd.notna(e50):
                    if e20 > e50:
                        if pos_open and pos_side == "short":
                            final_signal, reason = 1, "fallback_close_short"
                        elif not pos_open:
                            final_signal, reason = 1, "fallback_open_long"
                    elif e20 < e50:
                        if pos_open and pos_side == "long":
                            final_signal, reason = -1, "fallback_close_long"
                        elif not pos_open and st.session_state.allow_shorts:
                            final_signal, reason = -1, "fallback_open_short"

            # Execute trade
            if final_signal != 0 and st.session_state.auto_trade:
                update_trades(final_signal, last_px, now_ts)
                st.session_state.last_action_signal = final_signal

            st.session_state.last_decision = {"ts": now_ts, "signal": final_signal, "reason": reason, "conf": conf}
            st.session_state.decision_history = (st.session_state.decision_history + [{
                "timestamp": now_ts, "price": last_px,
                "model_pred_raw": None if pred is None else str(pred),
                "model_conf": conf, "final_signal": final_signal, "reason": reason,
            }])[-200:]

    # keep a compact preview of last few bars
    st.session_state.last_bars = (st.session_state.last_bars + _to_jsonable(bars)[-5:])[-10:]
    return processed

# ================= Connection settings =================
with st.expander("🔑 Connection Settings", expanded=True):
    st.session_state["exchange_code"] = st.text_input("Exchange Code (e.g., NSE)", value=st.session_state["exchange_code"])
    st.session_state["stock_code"]    = st.text_input("Stock Code (e.g., NIFTY 50)", value=st.session_state["stock_code"])
    st.session_state["stock_token"]   = st.text_input("Stock Token (optional)", value=st.session_state["stock_token"])

    # API key & secret are hard-coded above; we only ask for Session Token here
    session_token = st.text_input("Session Token", type="password", help="Paste your active Breeze session token")
    uploaded_model_file = st.file_uploader("Upload ML Model (.pkl)", type=["pkl"])

    c1, c2, c3 = st.columns(3)
    with c1: connect_pressed = st.button("🚀 Connect & Subscribe", disabled=(BreezeConnect is None))
    with c2: st.toggle("🔁 Auto-update charts", key="run_live", value=st.session_state.get("run_live", False))
    with c3: st.checkbox("Allow short selling", key="allow_shorts", value=st.session_state.get("allow_shorts", True))
    # heartbeat accessor
    try:
        hb = tickbus.heartbeat_value()
    except Exception:
        hb = getattr(tickbus, "heartbeat", lambda: 0)()
    st.caption(f"tickbus id: **{tickbus.BUS_ID}**  |  heartbeat: **{hb}**")

# ================= Grid search & seeding (optional) =================

def _normalize_grid_df(df: pd.DataFrame) -> pd.DataFrame:
    # standardize column names
    df = df.rename(columns={c: c.strip().replace(" ", "_").replace("-", "_") for c in df.columns})
    # flexible rename map to our expected names
    rename_map = {
        "ml_thresh": "ml_threshold",
        "ml_prob_threshold": "ml_threshold",
        "prob_threshold": "ml_threshold",
        "confidence_threshold": "ml_threshold",
        "ml_threshold": "ml_threshold",

        "adx_mult": "adx_target_mult",
        "adx_gate_mult": "adx_target_mult",
        "adx_target_mult": "adx_target_mult",

        "trail": "trail_mult",
        "trail_mult": "trail_mult",
        "trailing_mult": "trail_mult",

        "time_limit": "time_limit",
        "hold_minutes": "time_limit",
        "duration_limit": "time_limit",

        "pnl": "total_pnl",
        "total_pnl": "total_pnl",
        "net_pnl": "total_pnl",

        "max_dd": "max_drawdown",
        "max_drawdown": "max_drawdown",

        "trades": "trade_count",
        "trade_count": "trade_count",
    }
    for src, dst in rename_map.items():
        if src in df.columns and src != dst:
            df.rename(columns={src: dst}, inplace=True)
    # coerce numerics
    for c in ["ml_threshold","adx_target_mult","trail_mult","time_limit","total_pnl","max_drawdown","trade_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # sort preference
    sort_cols, ascending = [], []
    if "total_pnl" in df.columns:
        sort_cols.append("total_pnl"); ascending.append(False)
    if "max_drawdown" in df.columns:
        sort_cols.append("max_drawdown"); ascending.append(True)
    if "trade_count" in df.columns:
        sort_cols.append("trade_count"); ascending.append(False)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    return df

def _apply_row_to_session(row: pd.Series):
    # apply knobs if present
    if "ml_threshold"  in row and pd.notna(row["ml_threshold"]):   st.session_state.conf_threshold = float(row["ml_threshold"])
    if "adx_target_mult" in row and pd.notna(row["adx_target_mult"]): st.session_state.adx_target_mult = float(row["adx_target_mult"])
    if "trail_mult"     in row and pd.notna(row["trail_mult"]):    st.session_state.trail_mult = float(row["trail_mult"])
    if "time_limit"     in row and pd.notna(row["time_limit"]):    st.session_state.time_limit = int(row["time_limit"])

with st.expander("🧪 Grid Search & Seeding (optional)", expanded=True):
    colA, colB = st.columns(2)

    # ---- Left: Upload any grid CSV, normalize, preview, pick & apply ----
    with colA:
        grid_file = st.file_uploader("Upload grid_search CSV", type=["csv"], key="gsu_any")
        if grid_file is not None:
            try:
                gdf_raw = pd.read_csv(grid_file)
                gdf = _normalize_grid_df(gdf_raw.copy())
                show_cols = [c for c in ["ml_threshold","adx_target_mult","trail_mult","time_limit","total_pnl","max_drawdown","trade_count"] if c in gdf.columns]
                st.dataframe(gdf.head(12)[show_cols] if show_cols else gdf.head(12))
                idx = st.number_input("Pick a row to apply (0-based)", min_value=0, max_value=max(0, len(gdf)-1), value=0, step=1)
                if st.button("Apply selected row"):
                    _apply_row_to_session(gdf.iloc[int(idx)])
                    st.success(
                        f"Applied → conf ≥ {st.session_state.conf_threshold:.2f} | "
                        f"ADX × {st.session_state.adx_target_mult:.2f} | "
                        f"trail × {st.session_state.trail_mult:.2f} | "
                        f"time {st.session_state.time_limit}m"
                    )
            except Exception as e:
                st.error(f"Grid parse error: {e}")

        st.session_state.conf_threshold = st.slider("Model confidence threshold", 0.50, 0.95, float(st.session_state.conf_threshold), 0.01)
        st.session_state.adx_target_mult = st.slider("ADX gate multiplier", 0.0, 3.0, float(st.session_state.adx_target_mult), 0.1)
        st.session_state.time_limit = st.number_input("Max holding time (minutes, 0=disabled)", min_value=0, value=int(st.session_state.time_limit), step=1)
        st.checkbox("Enable Auto-Trade", key="auto_trade", value=st.session_state.get("auto_trade", True))

    # ---- Right: Auto-apply from normalized file if present ----
    with colB:
        st.caption("If you’ve already normalized your grid file, place it at `/mnt/data/grid_search_normalized.csv`.")
        norm_path = "/mnt/data/grid_search_normalized.csv"
        exists = os.path.exists(norm_path)
        st.write(f"Normalized file present: **{'Yes' if exists else 'No'}**")

        auto_apply = st.checkbox("Auto-apply best row on boot (from normalized file)", value=st.session_state.get("gs_auto_apply", True))
        st.session_state["gs_auto_apply"] = auto_apply
        if "gs_applied_once" not in st.session_state:
            st.session_state["gs_applied_once"] = False

        def _auto_apply_best_now():
            try:
                gdf_raw = pd.read_csv(norm_path)
                gdf = _normalize_grid_df(gdf_raw.copy())
                if not gdf.empty:
                    _apply_row_to_session(gdf.iloc[0])
                    st.session_state["gs_applied_once"] = True
                    st.success(
                        f"Auto-applied best row → conf ≥ {st.session_state.conf_threshold:.2f} | "
                        f"ADX × {st.session_state.adx_target_mult:.2f} | "
                        f"trail × {st.session_state.trail_mult:.2f} | "
                        f"time {st.session_state.time_limit}m"
                    )
                else:
                    st.info("Normalized grid file is empty after normalization.")
            except Exception as e:
                st.error(f"Auto-apply error: {e}")

        # auto-apply once per session if file exists and toggle is on
        if exists and auto_apply and not st.session_state["gs_applied_once"]:
            _auto_apply_best_now()

        if st.button("Apply best from normalized file now", disabled=not exists):
            _auto_apply_best_now()

    # ---- 1-minute OHLC seeding (unchanged, kept for faster warmup) ----
    st.markdown("---")
    seed_file = st.file_uploader("Upload 1-minute OHLC seed (CSV)", type=["csv"], key="seed_1m")
    st.caption("Expected columns: timestamp, open, high, low, close, (volume optional) — timestamps interpreted in IST.")
    if seed_file is not None:
        try:
            seed = pd.read_csv(seed_file)
            for c in list(seed.columns):
                if c.lower() == "datetime":
                    seed.rename(columns={c: "timestamp"}, inplace=True)
            seed["timestamp"] = pd.to_datetime(seed["timestamp"], errors="coerce")
            if getattr(seed["timestamp"].dt, "tz", None) is None:
                seed["timestamp"] = seed["timestamp"].dt.tz_localize(IST)
            seed = seed.dropna(subset=["timestamp"]).sort_values("timestamp")
            seed_df = pd.DataFrame({
                "timestamp": seed["timestamp"],
                "last_traded_price": pd.to_numeric(seed["close"], errors="coerce"),
                "volume": pd.to_numeric(seed.get("volume", 0), errors="coerce").fillna(0),
                "raw": "seed"
            }).dropna(subset=["last_traded_price"])
            st.session_state.live_data = (
                pd.concat([seed_df, st.session_state.live_data], ignore_index=True)
                .tail(MAX_WINDOW_SIZE).reset_index(drop=True)
            )
            st.session_state.live_data = calculate_indicators_live(st.session_state.live_data.copy())
            st.success(f"Seeded {len(seed_df)} bars → indicators ready sooner (ADX/BB/return_1h).")
        except Exception as e:
            st.error(f"Seed parse error: {e}")


# ================= Connect (UI thread only) =================
if connect_pressed and st.session_state.get("breeze") is None:
    if BreezeConnect is None:
        st.error("breeze-connect is not installed. `pip install breeze-connect`")
    elif not _keys_ok():
        pass  # error already shown
    else:
        exchange_code = st.session_state["exchange_code"]
        stock_code    = st.session_state["stock_code"]
        stock_token   = st.session_state["stock_token"]

        if not (session_token and exchange_code):
            st.error("⚠️ Provide session token and exchange code.")
        elif uploaded_model_file is None:
            st.error("⚠️ Upload your ML model file first (.pkl).")
        else:
            try:
                breeze = BreezeConnect(api_key=BREEZE_API_KEY)
                breeze.on_ticks = on_ticks
                breeze.generate_session(api_secret=BREEZE_API_SECRET, session_token=session_token)
                breeze.ws_connect()

                if stock_token.strip():
                    breeze.subscribe_feeds(stock_token=stock_token.strip(),
                                           get_market_depth=True, get_exchange_quotes=True)
                elif stock_code.strip():
                    breeze.subscribe_feeds(exchange_code=exchange_code, stock_code=stock_code.strip(),
                                           product_type="cash", get_market_depth=True, get_exchange_quotes=True)
                else:
                    raise ValueError("No instrument provided")

                st.session_state.breeze = breeze

                # load model
                model_bytes = uploaded_model_file.read()
                if joblib is None:
                    st.warning("joblib not installed; cannot load model. `pip install joblib`")
                else:
                    st.session_state.model = joblib.load(io.BytesIO(model_bytes))

                st.success("✅ Connected, subscribed & model loaded.")
                st.info(f"Connected with tickbus id **{tickbus.BUS_ID}**")

            except Exception as e:
                st.error(f"Connection error: {e}")
                lg.error(f"Connection error: {e}")

# ================= Debug tools =================
with st.expander("🔧 Debug tools"):
    if st.button("➕ Simulate 1s bar (no broker)"):
        # emit a few raw ticks within this second; then force-flush
        base = 100.0 + np.random.uniform(-0.3, 0.3)
        now = time.time()
        for j in range(5):
            tickbus.put_raw_tick({"ts": now + j*0.05, "price": base + np.random.normal(0, 0.05), "size": 1})
        # Force immediate emission (safe even if bucket empty)
        try:
            flushed = tickbus.flush_now()
        except Exception as e:
            flushed = False
            st.error(f"Flush error: {e}")
        st.success(f"Injected test ticks → flushed={flushed}")

# ================= Placeholders =================
ph_metrics = st.empty(); ph_candles = st.empty(); ph_line = st.empty()
ph_rsi = st.empty(); ph_atr = st.empty(); ph_table = st.empty()
ph_pos = st.empty(); ph_trades = st.empty(); ph_equity = st.empty()

# ================= Charts =================
def make_candles_with_signals(df_bars: pd.DataFrame, trades: list, current_pos: dict | None):
    if df_bars.empty: return go.Figure()
    d = df_bars.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    if getattr(d["timestamp"].dt, "tz", None) is None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(IST)
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Build 1-minute candles from 1-second bars for readability
    bars = (
        d.set_index("timestamp")
         .resample("1min")
         .agg(open=("last_traded_price","first"),
              high=("last_traded_price","max"),
              low =("last_traded_price","min"),
              close=("last_traded_price","last"),
              vol=("volume","sum"))
         .dropna(subset=["open","high","low","close"])
    )
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=bars.index, open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"], name="Price"))
    overlays = []
    if "ema_20" in d.columns:
        ema20 = d.set_index("timestamp")["ema_20"].dropna().reindex(bars.index, method="pad")
        overlays.append(("EMA 20", ema20))
    if "ema_50" in d.columns:
        ema50 = d.set_index("timestamp")["ema_50"].dropna().reindex(bars.index, method="pad")
        overlays.append(("EMA 50", ema50))
    for name, series in overlays:
        fig.add_trace(go.Scatter(x=bars.index, y=series, mode="lines", name=name))

    # Entry/exit markers (aligned to minute bars)
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for t in trades or []:
        et = pd.to_datetime(t["entry_time"], errors="coerce")
        xt = pd.to_datetime(t.get("exit_time"),  errors="coerce")
        if pd.notna(et):
            et_bar = bars.index.asof(et)
            if pd.notna(et_bar): buy_x.append(et_bar); buy_y.append(bars.loc[et_bar, "open"])
        if pd.notna(xt):
            xt_bar = bars.index.asof(xt)
            if pd.notna(xt_bar): sell_x.append(xt_bar); sell_y.append(bars.loc[xt_bar, "close"])
    if current_pos is not None:
        et = pd.to_datetime(current_pos["entry_time"], errors="coerce")
        if pd.notna(et):
            et_bar = bars.index.asof(et)
            if pd.notna(et_bar): buy_x.append(et_bar); buy_y.append(bars.loc[et_bar, "open"])

    if buy_x:
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers", name="BUY/COVER",
                                 marker=dict(symbol="triangle-up", size=12)))
    if sell_x:
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers", name="SELL/SHORT",
                                 marker=dict(symbol="triangle-down", size=12)))
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=True)
    return fig

# ================= Render once =================
def render_dashboard_once():
    processed = process_bar_queue()
    df = st.session_state.live_data.copy()
    if df.empty:
        ph_table.info("⚙️ Connected? Wait for live bars…")
        return processed

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(df["timestamp"].dt, "tz", None) is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
    df["last_traded_price"] = pd.to_numeric(df["last_traded_price"], errors="coerce").ffill()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if len(df) == 1:
        pad = df.iloc[-1].copy()
        pad["timestamp"] = pad["timestamp"] - pd.to_timedelta(1, unit="s")
        df = pd.concat([pd.DataFrame([pad]), df], ignore_index=True)

    latest_price = float(df["last_traded_price"].iloc[-1])
    # mark open pnl
    open_pnl = 0.0
    if st.session_state.position is not None:
        side = st.session_state.position["side"]
        entry = float(st.session_state.position["entry_price"])
        open_pnl = (latest_price - entry) if side == "long" else (entry - latest_price)
    total_pnl = float(sum(t['pnl'] for t in st.session_state.trades))

    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📈 Last Price", f"₹{latest_price:.2f}")
        c2.metric("💰 Open PnL", f"{open_pnl:.2f}")
        c3.metric("📊 Total PnL", f"{total_pnl:.2f}")
        try:
            hb = tickbus.heartbeat_value()
        except Exception:
            hb = getattr(tickbus, "heartbeat", lambda: 0)()
        c4.metric("🫀 Heartbeat", hb)

    with ph_candles.container():
        st.subheader("📊 Candles + Signals (IST)")
        st.plotly_chart(make_candles_with_signals(df, st.session_state.trades, st.session_state.position),
                        use_container_width=True)

    with st.expander("🛠 Live Debug", expanded=True):
        st.write(f"Heartbeat: **{hb}**")
        st.write(f"Live rows: **{len(st.session_state.live_data)}**")
        st.write(f"tickbus id: **{tickbus.BUS_ID}**")
        if not st.session_state.live_data.empty:
            tail = st.session_state.live_data.tail(5).copy()
            tail["timestamp"] = pd.to_datetime(tail["timestamp"], errors="coerce")
            if getattr(tail["timestamp"].dt, "tz", None) is None:
                tail["timestamp"] = tail["timestamp"].dt.tz_localize(IST)
            st.dataframe(tail[["timestamp","last_traded_price","volume"]])
        st.write("Last bar payloads:")
        if st.session_state.last_bars:
            try: st.json(st.session_state.last_bars[-1])
            except Exception: st.code(json.dumps(_to_jsonable(st.session_state.last_bars[-1]), indent=2))
        else:
            st.write("—")
        if st.button("Force refresh now"): st.rerun()

    with ph_line.container():
        st.subheader("📉 Price & Volume (1s bars)")
        st.line_chart(df.set_index("timestamp")[["last_traded_price", "volume"]])

    if "RSI" in df.columns:
        with ph_rsi.container():
            st.subheader("🔄 RSI")
            st.line_chart(df.set_index("timestamp")[["RSI"]])

    if "ATR" in df.columns:
        with ph_atr.container():
            st.subheader("📊 ATR")
            atr_df = df[["timestamp","ATR"]].dropna()
            if not atr_df.empty: st.line_chart(atr_df.set_index("timestamp")[["ATR"]])
            else: st.info("ATR warming up…")

    with ph_table.container():
        st.subheader("📝 Latest Bars (IST)")
        st.dataframe(df.tail(30))
        st.subheader("📦 Raw Bar Payloads")
        st.dataframe(st.session_state.live_data.tail(30)[["timestamp","raw"]])
        cdl, cdr = st.columns(2)
        with cdl:
            st.download_button("⬇️ Download Cleaned CSV", data=df.to_csv(index=False), file_name="live_cleaned.csv", mime="text/csv")
        with cdr:
            raw_df = st.session_state.live_data.copy()
            st.download_button("⬇️ Download Raw Bars CSV", data=raw_df[["timestamp","raw"]].to_csv(index=False), file_name="live_raw_bars.csv", mime="text/csv")

    with ph_pos.container():
        pos = st.session_state.position
        st.info(
            f"🟢 Open {pos['side'].upper()} | Entry ₹{pos['entry_price']:.2f} at {pos['entry_time']}"
            if pos else "⚪ No open position"
        )

    with ph_trades.container():
        if st.session_state.trades:
            st.subheader("📑 Closed Trades")
            st.dataframe(pd.DataFrame(st.session_state.trades))

    with ph_equity.container():
        if st.session_state.equity_curve:
            st.subheader("📈 Equity Curve (Total PnL)")
            eq = pd.DataFrame(st.session_state.equity_curve)
            eq["timestamp"] = pd.to_datetime(eq["timestamp"], errors="coerce")
            if getattr(eq["timestamp"].dt, "tz", None) is None:
                eq["timestamp"] = eq["timestamp"].dt.tz_localize(IST)
            eq = eq.dropna(subset=["timestamp"]).sort_values("timestamp")
            st.line_chart(eq.set_index("timestamp")["total_pnl"])

    return processed

# ================= Live render loop =================
processed_rows = render_dashboard_once()
now = time.time()
should_rerun = False
if processed_rows and processed_rows > 0:
    should_rerun = True
elif st.session_state.get("run_live", False) and st.session_state.get("breeze") is not None:
    if now - st.session_state.last_render_ts >= IDLE_REFRESH_SEC:
        should_rerun = True
if should_rerun:
    st.session_state.last_render_ts = now
    time.sleep(RENDER_SLEEP_SEC)
    st.rerun()

# ================= Raw preview + log DL =================
st.subheader("🟢 Last 10 Bar Previews")
if st.session_state.last_bars:
    safe = _to_jsonable(st.session_state.last_bars[-10:])
    try: st.json(safe)
    except Exception: st.code(json.dumps(safe, indent=2))
else:
    st.write("⚙️ Waiting for bars…")

if st.button("📥 Download Logs"):
    try:
        with open("live_trading.log", "r") as f:
            st.download_button("Download log file", f.read(), "live_trading.log", "text/plain")
    except Exception as e:
        st.error(f"Log read error: {e}")