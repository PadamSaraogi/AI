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
import streamlit as st
import pandas as pd
import ta
import sys
import joblib
import time
import io

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
    
            # Convert to datetime if needed
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
    
            if portfolio_equity is not None and len(portfolio_equity) > 1:
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
                ax.hist(wins, bins=range(1, max(wins)+2), alpha=0.7, label='Winning Streaks', color='green')
                ax.hist(losses, bins=range(1, max(losses)+2), alpha=0.7, label='Losing Streaks', color='red')
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

    st.title("📊 Live Trading Dashboard")

    # ========= Pure module-level BUS (no Streamlit here) =========
    try:
        BUS
    except NameError:
        BUS = {
            "q": queue.Queue(),          # tick queue
            "seq": itertools.count(1),   # micro timestamp bump
            "hb": 0,                     # heartbeat counter
        }

    # ========= Constants & logging =========
    MAX_WINDOW_SIZE  = 15000
    RENDER_SLEEP_SEC = 0.5
    IDLE_REFRESH_SEC = 3.0

    logger = logging.getLogger("LiveTradingLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("live_trading.log")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    # ========= Session defaults (UI thread only) =========
    defaults = {
        "live_data": pd.DataFrame(),
        "position": None,
        "trades": [],
        "equity_curve": [],
        "model": None,
        "breeze": None,
        "last_ticks": [],
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ========= Utilities =========
    def _clean_num(x):
        if x is None: return np.nan
        if isinstance(x, str): x = x.replace(",", "").strip()
        return pd.to_numeric(x, errors="coerce")

    def _parse_ts(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return pd.NaT
        ts = pd.to_datetime(v, errors="coerce", utc=True)
        if pd.notna(ts): return ts
        try:
            x = float(v)
            unit = "ms" if x > 1e10 else "s"
            return pd.to_datetime(x, unit=unit, utc=True, errors="coerce")
        except Exception:
            return pd.NaT

    def _to_jsonable(obj):
        """Recursively convert to JSON-safe; else string."""
        try:
            json.dumps(obj)
            return obj
        except Exception:
            pass
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(x) for x in obj]
        try:
            import numpy as _np
            if isinstance(obj, _np.generic):
                return obj.item()
        except Exception:
            pass
        try:
            import pandas as _pd
            if isinstance(obj, _pd.Timestamp):
                return obj.isoformat()
        except Exception:
            pass
        return str(obj)

    # ========= Tick parsing (no st.* here) =========
    def _extract_ts_price(t):
        # normalize to dict
        if isinstance(t, str):
            try: t = json.loads(t)
            except Exception: t = {"raw": t}

        # unwrap common nesting
        if isinstance(t, dict):
            for k in ("data", "payload", "tick", "d"):
                if isinstance(t.get(k), dict):
                    t = t[k]; break

        # timestamp
        ts_raw = (
            t.get("ltt") or t.get("last_trade_time") or t.get("exchange_time")
            or t.get("trade_time") or t.get("time") or t.get("timestamp")
            or t.get("datetime") or t.get("created_at") or None
        )
        ts = _parse_ts(ts_raw)
        if pd.isna(ts):
            ts = pd.to_datetime(time.time_ns(), unit="ns", utc=True, errors="coerce")

        # unique micro-bump (numeric timedelta)
        try:
            seq = next(BUS["seq"])
            ts = ts + pd.to_timedelta(seq % 1_000_000, unit="us")
        except Exception:
            pass

        # price
        price_keys = [
            "last","Last","LAST","last_traded_price","LastTradedPrice","lastTradedPrice",
            "ltp","LTP","lastPrice","LastPrice","close","Close","price","Price"
        ]
        ltp = None
        if isinstance(t, dict):
            for k in price_keys:
                if k in t and t[k] not in (None, ""):
                    ltp = t[k]; break
            if ltp is None:
                md = t.get("md") or t.get("market_depth") or {}
                bid = md.get("best_bid_price") or md.get("bid") or (md.get("b", {}) or {}).get("p")
                ask = md.get("best_ask_price") or md.get("ask") or (md.get("a", {}) or {}).get("p")
                bid = _clean_num(bid); ask = _clean_num(ask)
                if not np.isnan(bid) and not np.isnan(ask):
                    ltp = (bid + ask) / 2.0
            if ltp is None:
                last_trade = t.get("last_trade") or {}
                ltp = last_trade.get("price")
        ltp = _clean_num(ltp)
        return ts, ltp, t

    # ========= Indicators / features =========
    def _compute_feature_block(df_ticks: pd.DataFrame) -> pd.DataFrame:
        if df_ticks.empty: return df_ticks
        d = df_ticks.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
        d = d.dropna(subset=["timestamp"]).sort_values("timestamp")

        bars = (
            d.set_index("timestamp")
             .resample("1min")
             .agg(open=("last_traded_price","first"),
                  high=("last_traded_price","max"),
                  low =("last_traded_price","min"),
                  close=("last_traded_price","last"),
                  vol  =("volume","sum"))
             .dropna(subset=["open","high","low","close"])
        )
        if bars.empty:
            d["hour_of_day"] = d["timestamp"].dt.hour.astype("float64")
            vol = pd.to_numeric(d["volume"], errors="coerce")
            d["volume_spike_ratio"] = (vol / vol.rolling(200, min_periods=20).mean()).astype("float64")
            return d

        try:
            adx = ta.trend.adx(bars["high"], bars["low"], bars["close"], window=14)
            bb  = ta.volatility.BollingerBands(bars["close"], window=20, window_dev=2)
            bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        except Exception:
            adx = pd.Series(index=bars.index, dtype=float)
            bb_width = pd.Series(index=bars.index, dtype=float)

        ret_1h = (bars["close"] / bars["close"].shift(60) - 1.0)

        feat_cols = ["ADX14","bb_width","return_1h"]
        feat = pd.DataFrame({"ADX14": adx, "bb_width": bb_width, "return_1h": ret_1h}).sort_index()

        # drop existing to avoid suffixes
        d = d.drop(columns=[c for c in feat_cols if c in d.columns], errors="ignore")

        d = pd.merge_asof(
            d.sort_values("timestamp"),
            feat.dropna(how="all"),
            left_on="timestamp",
            right_index=True,
            direction="backward",
            allow_exact_matches=True,
        )

        d["hour_of_day"] = d["timestamp"].dt.hour.astype("float64")
        vol = pd.to_numeric(d["volume"], errors="coerce")
        d["volume_spike_ratio"] = (vol / vol.rolling(200, min_periods=20).mean()).astype("float64")
        return d

    def calculate_indicators_live(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        d = df.copy().sort_values("timestamp")
        d["last_traded_price"] = pd.to_numeric(d["last_traded_price"], errors="coerce").ffill()
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)

        for base in ["ADX14","bb_width","return_1h"]:
            if base not in d.columns and f"{base}_x" in d.columns:
                d.rename(columns={f"{base}_x": base}, inplace=True)
            for suf in ("_x","_y"):
                col = f"{base}{suf}"
                if col in d.columns and col != base:
                    d.drop(columns=[col], inplace=True, errors="ignore")

        price = d["last_traded_price"]
        if len(d) >= 20: d["ema_20"] = ta.trend.ema_indicator(price, window=20)
        if len(d) >= 50: d["ema_50"] = ta.trend.ema_indicator(price, window=50)
        if len(d) >= 14:
            d["ATR"] = ta.volatility.average_true_range(high=price, low=price, close=price, window=14)
            d["RSI"] = ta.momentum.rsi(price, window=14)

        d = _compute_feature_block(d)
        return d

    # ========= Model prediction =========
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
                try: buy_idx = list(classes_).index(1)
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

    # ========= Trading state updates =========
    def update_trades(signal, price, timestamp):
        pos = st.session_state.position
        if signal == 1 and pos is None:
            st.session_state.position = {"entry_price": float(price), "entry_time": timestamp}
            logger.info(f"Open LONG @ {price} on {timestamp}")
        elif signal == -1 and pos is not None:
            pnl = float(price) - float(pos["entry_price"])
            st.session_state.trades.append({
                "entry_price": float(pos["entry_price"]),
                "exit_price": float(price),
                "entry_time": pos["entry_time"],
                "exit_time": timestamp,
                "pnl": float(pnl),
            })
            st.session_state.position = None
            logger.info(f"Close LONG @ {price} on {timestamp} | PnL {pnl:.2f}")
        total = sum(t['pnl'] for t in st.session_state.trades)
        if st.session_state.position is not None:
            total += (float(price) - float(st.session_state.position["entry_price"]))
        st.session_state.equity_curve.append({"timestamp": timestamp, "total_pnl": float(total)})

    # ========= WS callback (NO st.* calls) =========
    def on_ticks(ticks):
        try:
            BUS["q"].put(ticks, block=False)
        except Exception:
            pass
        try:
            BUS["hb"] = (BUS.get("hb", 0) + 1) % 1_000_000_000
        except Exception:
            BUS["hb"] = 0
        logger.info(f"Tick: {ticks}")

    # ========= Queue processor (UI thread) =========
    def process_tick_queue():
        processed = 0
        q = BUS["q"]
        while not q.empty():
            ticks = q.get()
            st.session_state.last_ticks.append(_to_jsonable(ticks))  # keep preview safe
            st.session_state.last_ticks = st.session_state.last_ticks[-10:]
            batch = ticks if isinstance(ticks, list) else [ticks]

            rows = []
            for t in batch:
                ts, ltp, raw = _extract_ts_price(t)
                vol = None
                if isinstance(t, dict):
                    vol = t.get("volume") or t.get("ltq") or t.get("quantity")
                if pd.isna(ltp) and not st.session_state.live_data.empty:
                    ltp = pd.to_numeric(st.session_state.live_data["last_traded_price"], errors="coerce").ffill().iloc[-1]
                rows.append({
                    "timestamp": ts,
                    "last_traded_price": _clean_num(ltp),
                    "volume": _clean_num(vol),
                    "raw": json.dumps(raw) if isinstance(raw, dict) else str(raw),
                })

            if rows:
                df_new = pd.DataFrame(rows)
                # repair any NaT timestamps so nothing gets dropped later
                if "timestamp" in df_new.columns:
                    bad_ts = df_new["timestamp"].isna()
                    if bad_ts.any():
                        base = (st.session_state.live_data["timestamp"].iloc[-1]
                                if ("timestamp" in st.session_state.live_data.columns and
                                    not st.session_state.live_data.empty)
                                else pd.Timestamp.utcnow().tz_localize("UTC"))
                        fixes = []
                        for i, is_bad in enumerate(bad_ts):
                            if not is_bad:
                                fixes.append(df_new.loc[i, "timestamp"])
                            else:
                                base = base + pd.to_timedelta(1, unit="us")
                                fixes.append(base)
                        df_new["timestamp"] = fixes

                st.session_state.live_data = (
                    pd.concat([st.session_state.live_data, df_new], ignore_index=True)
                    .tail(MAX_WINDOW_SIZE)
                    .reset_index(drop=True)
                )
                processed += len(df_new)

        if processed > 0:
            df = calculate_indicators_live(st.session_state.live_data.copy())
            st.session_state.live_data = df

            # model + decisions
            if st.session_state.model is not None and not df.empty:
                pred, conf = predict_signal(st.session_state.model, df)
                sig = _norm_signal(pred)

                def adx_gate_ok(dff):
                    if "ADX14" not in dff.columns or st.session_state.adx_target_mult <= 0:
                        return True
                    adx_series = pd.to_numeric(dff["ADX14"], errors="coerce").dropna()
                    if len(adx_series) < 20: return True
                    cur = float(adx_series.iloc[-1])
                    base = float(adx_series.tail(200).median())
                    return cur >= st.session_state.adx_target_mult * base

                def time_limit_exit(now_ts):
                    tl = int(st.session_state.time_limit or 0)
                    if tl <= 0 or st.session_state.position is None: return False
                    held = (now_ts - pd.to_datetime(st.session_state.position["entry_time"])).total_seconds() / 60
                    return held >= tl

                now_ts = pd.to_datetime(df["timestamp"].iloc[-1])
                last_px = float(pd.to_numeric(df["last_traded_price"].iloc[-1], errors="coerce"))

                final_signal, reason = 0, "none"
                if time_limit_exit(now_ts):
                    final_signal = -1 if st.session_state.position is not None else 0
                    reason = "time_limit"
                elif sig != 0 and conf is not None and conf >= float(st.session_state.conf_threshold) and adx_gate_ok(df):
                    if sig != st.session_state.last_action_signal:
                        final_signal = sig; reason = "model_conf_ok"
                elif {"ema_20","ema_50"}.issubset(df.columns):
                    e20, e50 = df["ema_20"].iloc[-1], df["ema_50"].iloc[-1]
                    pos_open = st.session_state.position is not None
                    if pd.notna(e20) and pd.notna(e50):
                        if e20 > e50 and not pos_open:
                            final_signal = 1; reason = "fallback_ema"
                        elif e20 < e50 and pos_open:
                            final_signal = -1; reason = "fallback_ema"

                if final_signal != 0 and st.session_state.auto_trade:
                    update_trades(final_signal, last_px, now_ts)
                    st.session_state.last_action_signal = final_signal

                st.session_state.last_decision = {"ts": now_ts, "signal": final_signal, "reason": reason, "conf": conf}
                dh = st.session_state.decision_history
                dh.append({
                    "timestamp": now_ts, "price": last_px,
                    "model_pred_raw": None if pred is None else str(pred),
                    "model_conf": conf, "final_signal": final_signal, "reason": reason,
                })
                st.session_state.decision_history = dh[-200:]

        return processed

    # ========= Connection settings =========
    with st.expander("🔑 Connection Settings", expanded=True):
        api_key = "=4c730660p24@d03%65343MG909o217L"
        api_secret = "416D2gJdy064P7F7)s5e590J8I1692~7"
        session_token = st.text_input("Session Token", type="password")
        exchange_code = st.text_input("Exchange Code (e.g., NSE)")
        stock_code = st.text_input("Stock Code (e.g., NIFTY 50)")
        stock_token = st.text_input("Stock Token (optional)", value="")
        uploaded_model_file = st.file_uploader("Upload ML Model (.pkl)", type=["pkl"])
        c1, c2 = st.columns(2)
        with c1: connect_pressed = st.button("🚀 Connect & Subscribe")
        with c2: st.toggle("🔁 Auto-update charts (continuous refresh)", key="run_live", value=st.session_state.get("run_live", False))

    # ========= Grid search + seed (optional) =========
    with st.expander("🧪 Grid Search & Seeding (optional)", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            grid_file = st.file_uploader("Upload grid_search_results_*.csv", type=["csv"], key="gsu")
            if grid_file is not None:
                gdf = pd.read_csv(grid_file)
                rename_map = {
                    "ml_thresh":"ml_threshold","ml_prob_threshold":"ml_threshold","confidence_threshold":"ml_threshold",
                    "adx_mult":"adx_target_mult","adx_gate_mult":"adx_target_mult",
                    "trail":"trail_mult",
                    "duration_limit":"time_limit","hold_minutes":"time_limit",
                    "pnl":"total_pnl","max_dd":"max_drawdown","trades":"trade_count",
                }
                gdf = gdf.rename(columns={k:v for k,v in rename_map.items() if k in gdf.columns})
                sort_cols = ["total_pnl"] + (["max_drawdown"] if "max_drawdown" in gdf.columns else [])
                sort_asc  = [False] + ([True] if "max_drawdown" in gdf.columns else [])
                gdf_sorted = gdf.sort_values(sort_cols, ascending=sort_asc, na_position="last").reset_index(drop=True)
                show_cols = [c for c in ["ml_threshold","adx_target_mult","trail_mult","time_limit","total_pnl","max_drawdown","trade_count"] if c in gdf_sorted.columns]
                st.dataframe(gdf_sorted.head(10)[show_cols])
                idx = st.selectbox("Pick a row to apply", options=range(min(10, len(gdf_sorted))), format_func=lambda i: f"Row {i}")
                best = gdf_sorted.iloc[int(idx)]
                if "ml_threshold" in best: st.session_state.conf_threshold = float(best["ml_threshold"])
                if "adx_target_mult" in best: st.session_state.adx_target_mult = float(best["adx_target_mult"])
                if "trail_mult" in best: st.session_state.trail_mult = float(best["trail_mult"])
                if "time_limit" in best: st.session_state.time_limit = int(best["time_limit"])
                st.success(f"Applied → conf ≥ {st.session_state.conf_threshold:.2f} | ADX × {st.session_state.adx_target_mult:.2f} | trail × {st.session_state.trail_mult:.2f} | time {st.session_state.time_limit}m")
            # manual knobs
            st.session_state.conf_threshold = st.slider("Model confidence threshold", 0.50, 0.95, float(st.session_state.conf_threshold), 0.01)
            st.session_state.adx_target_mult = st.slider("ADX gate multiplier", 0.0, 3.0, float(st.session_state.adx_target_mult), 0.1)
            st.session_state.time_limit = st.number_input("Max holding time (minutes, 0=disabled)", min_value=0, value=int(st.session_state.time_limit), step=1)
            st.checkbox("Enable Auto-Trade", key="auto_trade", value=st.session_state.get("auto_trade", True))
        with colB:
            seed_file = st.file_uploader("Upload 1-minute OHLC seed (CSV)", type=["csv"], key="seed")
            st.caption("Expected columns: timestamp, open, high, low, close, (volume optional)")
            if seed_file is not None:
                seed = pd.read_csv(seed_file)
                for c in list(seed.columns):
                    if c.lower() == "datetime": seed.rename(columns={c: "timestamp"}, inplace=True)
                seed["timestamp"] = pd.to_datetime(seed["timestamp"], utc=True, errors="coerce")
                seed = seed.dropna(subset=["timestamp"]).sort_values("timestamp")
                seed_df = pd.DataFrame({
                    "timestamp": seed["timestamp"],
                    "last_traded_price": pd.to_numeric(seed["close"], errors="coerce"),
                    "volume": pd.to_numeric(seed.get("volume", 0), errors="coerce").fillna(0),
                    "raw": "seed"
                }).dropna(subset=["last_traded_price"])
                st.session_state.live_data = pd.concat([seed_df, st.session_state.live_data], ignore_index=True).tail(MAX_WINDOW_SIZE).reset_index(drop=True)
                st.session_state.live_data = calculate_indicators_live(st.session_state.live_data.copy())
                st.success(f"Seeded {len(seed_df)} bars → indicators ready sooner (ADX/BB/return_1h).")

    # ========= Connect (UI thread only) =========
    if connect_pressed and st.session_state.get("breeze") is None:
        if not (api_key and api_secret and session_token and exchange_code):
            st.error("⚠️ Provide API key, secret, session token, and exchange code.")
        elif uploaded_model_file is None:
            st.error("⚠️ Upload your ML model file first (.pkl).")
        else:
            try:
                breeze = BreezeConnect(api_key=api_key)
                breeze.on_ticks = on_ticks          # callback has NO st.* calls
                breeze.generate_session(api_secret=api_secret, session_token=session_token)
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

                model_bytes = uploaded_model_file.read()
                st.session_state.model = joblib.load(io.BytesIO(model_bytes))
                st.success("✅ Connected, subscribed & model loaded.")
            except Exception as e:
                st.error(f"Connection error: {e}")
                logger.error(f"Connection error: {e}")

    # ========= Placeholders =========
    ph_metrics = st.empty()
    ph_candles = st.empty()
    ph_line = st.empty()
    ph_rsi = st.empty()
    ph_atr = st.empty()
    ph_table = st.empty()
    ph_pos = st.empty()
    ph_trades = st.empty()
    ph_equity = st.empty()

    # ========= Charts =========
    def make_candles_with_signals(df_ticks: pd.DataFrame, trades: list, current_pos: dict | None):
        if df_ticks.empty: return go.Figure()
        d = df_ticks.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
        d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
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
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        for t in trades or []:
            et = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
            xt = pd.to_datetime(t["exit_time"],  utc=True, errors="coerce")
            if pd.notna(et):
                et_bar = bars.index.asof(et)
                if pd.notna(et_bar): buy_x.append(et_bar); buy_y.append(bars.loc[et_bar, "open"])
            if pd.notna(xt):
                xt_bar = bars.index.asof(xt)
                if pd.notna(xt_bar): sell_x.append(xt_bar); sell_y.append(bars.loc[xt_bar, "close"])
        if current_pos is not None:
            et = pd.to_datetime(current_pos["entry_time"], utc=True, errors="coerce")
            if pd.notna(et):
                et_bar = bars.index.asof(et)
                if pd.notna(et_bar): buy_x.append(et_bar); buy_y.append(bars.loc[et_bar, "open"])
        if buy_x:
            fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=12)))
        if sell_x:
            fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=12)))
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=True)
        return fig

    # ========= Render once =========
    def render_dashboard_once():
        processed = process_tick_queue()

        df = st.session_state.live_data.copy()
        if df.empty:
            ph_table.info("⚙️ Connected? Wait for live ticks…")
            return processed

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["last_traded_price"] = pd.to_numeric(df["last_traded_price"], errors="coerce").ffill()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        if len(df) == 1:
            pad = df.iloc[-1].copy()
            pad["timestamp"] = pad["timestamp"] - pd.to_timedelta(1, unit="s")
            df = pd.concat([pd.DataFrame([pad]), df], ignore_index=True)

        latest_price = float(df["last_traded_price"].iloc[-1])
        open_pnl = (latest_price - float(st.session_state.position["entry_price"])) if st.session_state.position else 0.0
        total_pnl = float(sum(t['pnl'] for t in st.session_state.trades))

        with ph_metrics.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📈 Last Price", f"₹{latest_price:.2f}")
            c2.metric("💰 Open PnL", f"{open_pnl:.2f}")
            c3.metric("📊 Total PnL", f"{total_pnl:.2f}")
            c4.metric("🫀 Tick heartbeat", BUS.get("hb", 0))

        with ph_candles.container():
            st.subheader("📊 Candles + Signals")
            st.plotly_chart(make_candles_with_signals(df, st.session_state.trades, st.session_state.position), use_container_width=True)

        with st.expander("🛠 Live Debug", expanded=True):
            st.write(f"Queue size: **{BUS['q'].qsize()}**")
            st.write(f"Heartbeat: **{BUS.get('hb', 0)}**")
            st.write(f"Live rows: **{len(st.session_state.live_data)}**")
            if not st.session_state.live_data.empty:
                tail = st.session_state.live_data.tail(5).copy()
                tail["timestamp"] = pd.to_datetime(tail["timestamp"], utc=True, errors="coerce")
                st.dataframe(tail[["timestamp","last_traded_price","volume"]])
            st.write("Last raw tick batch:")
            if st.session_state.last_ticks:
                try:
                    st.json(st.session_state.last_ticks[-1])
                except Exception:
                    st.code(json.dumps(_to_jsonable(st.session_state.last_ticks[-1]), indent=2))
            else:
                st.write("—")
            if st.button("Force refresh now"):
                st.rerun()

        with ph_line.container():
            st.subheader("📉 Price & Volume")
            st.line_chart(df.set_index("timestamp")[["last_traded_price", "volume"]])

        if "RSI" in df.columns:
            with ph_rsi.container():
                st.subheader("🔄 RSI")
                st.line_chart(df.set_index("timestamp")[["RSI"]])

        if "ATR" in df.columns:
            with ph_atr.container():
                st.subheader("📊 ATR")
                atr_df = df[["timestamp","ATR"]].dropna()
                if not atr_df.empty:
                    st.line_chart(atr_df.set_index("timestamp")[["ATR"]])
                else:
                    st.info("ATR warming up…")

        with ph_table.container():
            st.subheader("📝 Latest Cleaned Data")
            st.dataframe(df.tail(30))
            st.subheader("📦 Raw Payloads")
            st.dataframe(st.session_state.live_data.tail(30)[["timestamp","raw"]])
            cdl, cdr = st.columns(2)
            with cdl:
                st.download_button("⬇️ Download Cleaned CSV", data=df.to_csv(index=False), file_name="live_cleaned.csv", mime="text/csv")
            with cdr:
                raw_df = st.session_state.live_data.copy()
                st.download_button("⬇️ Download Raw Ticks CSV", data=raw_df[["timestamp","raw"]].to_csv(index=False), file_name="live_raw_ticks.csv", mime="text/csv")

        with ph_pos.container():
            st.info(f"🟢 Open Position: Entry ₹{st.session_state.position['entry_price']:.2f} at {st.session_state.position['entry_time']}"
                    if st.session_state.position else "⚪ No open position")

        with ph_trades.container():
            if st.session_state.trades:
                st.subheader("📑 Closed Trades")
                st.dataframe(pd.DataFrame(st.session_state.trades))

        with ph_equity.container():
            if st.session_state.equity_curve:
                st.subheader("📈 Equity Curve (Total PnL)")
                eq = pd.DataFrame(st.session_state.equity_curve)
                eq["timestamp"] = pd.to_datetime(eq["timestamp"], errors="coerce", utc=True)
                eq = eq.dropna(subset=["timestamp"]).sort_values("timestamp")
                st.line_chart(eq.set_index("timestamp")["total_pnl"])

        return processed

    # ========= Live render loop (rerun when data arrives) =========
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

    # ========= Raw preview + log download (SAFE) =========
    st.subheader("🟢 Raw Tick Preview (last 10)")
    if st.session_state.last_ticks:
        safe = _to_jsonable(st.session_state.last_ticks[-10:])
        try:
            st.json(safe)
        except Exception:
            st.code(json.dumps(safe, indent=2))
    else:
        st.write("⚙️ Waiting for ticks…")

    if st.button("📥 Download Logs"):
        try:
            with open("live_trading.log", "r") as f:
                st.download_button("Download log file", f.read(), "live_trading.log", "text/plain")
        except Exception as e:
            st.error(f"Log read error: {e}")
