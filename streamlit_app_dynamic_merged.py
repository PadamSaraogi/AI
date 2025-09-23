
import os
import streamlit as st
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
    
    # ---------------- Constants ----------------
    MAX_WINDOW_SIZE = 15000            # rolling window for in-memory charting
    RENDER_LOOP_SECONDS = 20         # how long each live render loop runs per run
    RENDER_SLEEP_SEC = 1.0           # refresh interval within the loop
    FEATURES = ['ema_20', 'ema_50', 'ATR', 'RSI']
    
    # ---------------- Logging ----------------
    logger = logging.getLogger("LiveTradingLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("live_trading.log")
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    
    # ---------------- Thread-safe tick queue ----------------
    tick_queue = queue.Queue()
    
    # ---------------- Session State Defaults ----------------
    defaults = {
        "live_data": pd.DataFrame(),
        "position": None,            # {"entry_price": float, "entry_time": ts}
        "trades": [],                # list of {entry_price, exit_price, entry_time, exit_time, pnl}
        "equity_curve": [],          # list of {timestamp, total_pnl}
        "model": None,
        "breeze": None,
        "last_ticks": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # ---------------- Utils ----------------
    def calculate_indicators_live(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute EMA20, EMA50, ATR (on LTP proxy), RSI on the cleaned, time-sorted DF.
        """
        if df.empty or len(df) < 20:
            return df
    
        work = df.copy()
        work = work.set_index("timestamp")
        price = work['last_traded_price']
    
        work['ema_20'] = ta.trend.ema_indicator(price, window=20)
        work['ema_50'] = ta.trend.ema_indicator(price, window=50)
        # ATR proxy using LTP for H/L/C (we don't have real OHLC per tick)
        work['ATR'] = ta.volatility.average_true_range(
            high=price, low=price, close=price, window=14
        )
        work['RSI'] = ta.momentum.rsi(price, window=14)
    
        work = work.reset_index()
        return work
    
    def predict_signal(model, df: pd.DataFrame):
        """
        Return (pred, proba) given the most recent row with all FEATURES present.
        """
        if model is None:
            return None, None
        if not all(f in df.columns for f in FEATURES):
            return None, None
    
        latest = df.dropna(subset=FEATURES)
        if latest.empty:
            return None, None
    
        latest = latest.iloc[-1:][FEATURES]
        pred = model.predict(latest)[0]
        proba = float(getattr(model, "predict_proba", lambda X: np.array([[0, 1.0]]))(latest).max())
        return pred, proba
    
    def update_trades(signal, price, timestamp):
        """
        Basic long-only toggle: 1 = open if flat; -1 = close if open.
        Tracks equity curve from realized + open PnL.
        """
        pos = st.session_state.position
    
        if signal == 1 and pos is None:
            st.session_state.position = {"entry_price": price, "entry_time": timestamp}
            logger.info(f"Opened position at {price} on {timestamp}")
    
        elif signal == -1 and pos is not None:
            pnl = price - pos["entry_price"]
            trade_record = {
                "entry_price": pos["entry_price"],
                "exit_price": price,
                "entry_time": pos["entry_time"],
                "exit_time": timestamp,
                "pnl": pnl,
            }
            st.session_state.trades.append(trade_record)
            st.session_state.position = None
            logger.info(f"Closed position at {price} on {timestamp} | PnL {pnl:.2f}")
    
        # Equity tracking (realized + open)
        total_pnl = sum(t['pnl'] for t in st.session_state.trades)
        if st.session_state.position is not None:
            total_pnl += (price - st.session_state.position["entry_price"])
        st.session_state.equity_curve.append({"timestamp": timestamp, "total_pnl": total_pnl})
    
    def on_ticks(ticks):
        """
        Breeze websocket callback. Just enqueue ticks; do not touch Streamlit widgets here.
        """
        tick_queue.put(ticks)
        logger.info(f"Received raw tick: {ticks}")
    
    def process_tick_queue():
        """
        Drain the queue -> append to live_data -> clean types -> trim window -> compute indicators -> maybe generate signal.
        """
        processed_rows = 0
        while not tick_queue.empty():
            ticks = tick_queue.get()
            st.session_state.last_ticks.append(ticks)
            if len(st.session_state.last_ticks) > 10:
                st.session_state.last_ticks = st.session_state.last_ticks[-10:]
    
            new_rows = []
            # Normalize list/dict payloads from the feed
            if isinstance(ticks, list):
                it = ticks
            else:
                it = [ticks]
    
            for t in it:
                ltt = t.get("ltt", pd.Timestamp.utcnow())  # last traded time or now
                ltp = (
                    t.get("last")
                    or t.get("last_traded_price")
                    or t.get("ltp")
                    or t.get("lastPrice")
                    or np.nan
                )
                vol = t.get("volume") or t.get("ltq") or np.nan
                new_rows.append(
                    {"timestamp": pd.to_datetime(ltt, errors="coerce"),
                     "last_traded_price": pd.to_numeric(ltp, errors="coerce"),
                     "volume": pd.to_numeric(vol, errors="coerce")}
                )
    
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                # drop rows missing critical fields
                new_df = new_df.dropna(subset=["timestamp", "last_traded_price"])
                if not new_df.empty:
                    st.session_state.live_data = pd.concat(
                        [st.session_state.live_data, new_df], ignore_index=True
                    )
                    processed_rows += len(new_df)
    
        if processed_rows > 0:
            # keep only the latest MAX_WINDOW_SIZE and sort by time
            df = st.session_state.live_data
            df = df.dropna(subset=["timestamp", "last_traded_price"])
            df = df.sort_values("timestamp").tail(MAX_WINDOW_SIZE).reset_index(drop=True)
            df = calculate_indicators_live(df)
            st.session_state.live_data = df
    
            # ML signal (if model is loaded)
            if st.session_state.model is not None and not df.empty:
                pred, conf = predict_signal(st.session_state.model, df)
                if pred is not None:
                    latest_price = float(df["last_traded_price"].iloc[-1])
                    latest_timestamp = pd.to_datetime(df["timestamp"].iloc[-1])
                    update_trades(pred, latest_price, latest_timestamp)
    
    # ---------------- Connection Settings ----------------
    with st.expander("🔑 Connection Settings", expanded=True):
        # Prefer secrets; fall back to user inputs
        api_key = st.secrets.get("breeze_api_key", os.getenv("BREEZE_API_KEY", ""))
        api_secret = st.secrets.get("breeze_api_secret", os.getenv("BREEZE_API_SECRET", ""))
        if not api_key or not api_secret:
            st.info("Tip: set keys in `.streamlit/secrets.toml` as `breeze_api_key` and `breeze_api_secret`.")
    
        api_key = st.text_input("API Key", value=api_key or "", type="password")
        api_secret = st.text_input("API Secret", value=api_secret or "", type="password")
    
        session_token = st.text_input("Session Token", type="password")
        exchange_code = st.text_input("Exchange Code (e.g., NSE)")
        stock_code = st.text_input("Stock Code (e.g., NIFTY 50)")
        stock_token = st.text_input("Stock Token (optional)", value="")
        uploaded_model_file = st.file_uploader("Upload ML Model (.pkl)", type=["pkl"])
    
        col_conn1, col_conn2 = st.columns([1, 1])
        with col_conn1:
            connect_pressed = st.button("🚀 Connect & Subscribe")
        with col_conn2:
            run_live = st.toggle("🔁 Auto-update charts (keeps refreshing for ~20s cycles)", value=True)
    
    # ---------------- Connect & Subscribe ----------------
    if connect_pressed:
        if not (api_key and api_secret and session_token and exchange_code):
            st.error("⚠️ Please provide API key, API secret, session token, and exchange code.")
        elif uploaded_model_file is None:
            st.error("⚠️ Upload your ML model file first (.pkl).")
        else:
            try:
                breeze = BreezeConnect(api_key=api_key)
                # Set callback BEFORE connecting/subscribing so we don't miss the first payload
                breeze.on_ticks = on_ticks
                breeze.generate_session(api_secret=api_secret, session_token=session_token)
                breeze.ws_connect()
    
                # Subscribe using token OR (exchange + stock_code)
                if stock_token.strip():
                    breeze.subscribe_feeds(
                        stock_token=stock_token.strip(),
                        get_market_depth=True,
                        get_exchange_quotes=True
                    )
                elif stock_code.strip():
                    breeze.subscribe_feeds(
                        exchange_code=exchange_code,
                        stock_code=stock_code.strip(),
                        product_type="cash",
                        get_market_depth=True,
                        get_exchange_quotes=True
                    )
                else:
                    st.error("⚠️ Provide either Stock Code or Stock Token to subscribe.")
                    raise ValueError("No instrument provided")
    
                st.session_state.breeze = breeze
    
                # Load model
                model_bytes = uploaded_model_file.read()
                st.session_state.model = joblib.load(io.BytesIO(model_bytes))
                st.success("✅ Connected, subscribed & model loaded.")
                logger.info("Connected & subscribed successfully.")
            except Exception as e:
                st.error(f"Connection error: {e}")
                logger.error(f"Connection error: {e}")
    
    # ---------------- Placeholders for dynamic UI ----------------
    ph_metrics = st.empty()
    ph_price_vol = st.empty()
    ph_ema = st.empty()
    ph_rsi = st.empty()
    ph_atr = st.empty()
    ph_table = st.empty()
    ph_pos = st.empty()
    ph_trades = st.empty()
    ph_equity = st.empty()
    
    def render_dashboard_once():
        """
        One pass: drain queue, recompute indicators/signals, and rerender all UI placeholders.
        """
        process_tick_queue()
    
        df = st.session_state.live_data.copy()
        if df.empty:
            ph_table.info("⚙️ Connect with valid credentials and wait for live ticks...")
            return
    
        # strict typing & sorting (safety for charts)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["last_traded_price"] = pd.to_numeric(df["last_traded_price"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["timestamp", "last_traded_price"]).sort_values("timestamp")
        if df.empty:
            ph_table.info("Awaiting valid price ticks...")
            return
    
        latest_price = float(df["last_traded_price"].iloc[-1])
        open_pnl = 0.0
        if st.session_state.position:
            open_pnl = latest_price - float(st.session_state.position["entry_price"])
        total_pnl = float(sum(t['pnl'] for t in st.session_state.trades))
    
        # --- metrics row ---
        with ph_metrics.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("📈 Last Price", f"₹{latest_price:.2f}")
            c2.metric("💰 Open PnL", f"{open_pnl:.2f}")
            c3.metric("📊 Total PnL", f"{total_pnl:.2f}")
    
        # --- charts ---
        with ph_price_vol.container():
            st.subheader("📉 Price & Volume")
            st.line_chart(df.set_index("timestamp")[["last_traded_price", "volume"]])
    
        if {"ema_20", "ema_50"}.issubset(df.columns):
            with ph_ema.container():
                st.subheader("📍 EMA 20 & EMA 50")
                st.line_chart(df.set_index("timestamp")[["ema_20", "ema_50"]])
        else:
            ph_ema.empty()
    
        if "RSI" in df.columns:
            with ph_rsi.container():
                st.subheader("🔄 RSI")
                st.line_chart(df.set_index("timestamp")[["RSI"]])
        else:
            ph_rsi.empty()
    
        if "ATR" in df.columns:
            with ph_atr.container():
                st.subheader("📊 ATR")
                st.line_chart(df.set_index("timestamp")[["ATR"]])
        else:
            ph_atr.empty()
    
        # --- Data snapshot ---
        with ph_table.container():
            st.subheader("📝 Latest Data")
            st.dataframe(df.tail(10))
    
        # --- Position & Trades ---
        with ph_pos.container():
            if st.session_state.position:
                st.info(
                    f"🟢 Open Position: Entry ₹{st.session_state.position['entry_price']:.2f} "
                    f"at {st.session_state.position['entry_time']}"
                )
            else:
                st.info("⚪ No open position")
    
        with ph_trades.container():
            if st.session_state.trades:
                st.subheader("📑 Closed Trades")
                st.dataframe(pd.DataFrame(st.session_state.trades))
    
        # --- Equity Curve ---
        with ph_equity.container():
            if st.session_state.equity_curve:
                st.subheader("📈 Equity Curve (Total PnL)")
                eq_df = pd.DataFrame(st.session_state.equity_curve)
                eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], errors="coerce")
                eq_df = eq_df.dropna(subset=["timestamp"]).sort_values("timestamp")
                st.line_chart(eq_df.set_index("timestamp")["total_pnl"])
    
    # ---------------- Live render loop ----------------
    if run_live:
        start = time.time()
        while True:
            render_dashboard_once()
            time.sleep(RENDER_SLEEP_SEC)
            # Stop after a short burst so Streamlit can yield control & you can toggle again on next run
            if time.time() - start > RENDER_LOOP_SECONDS:
                break
    else:
        render_dashboard_once()
    
    # ---------------- Raw ticks preview & logs ----------------
    st.subheader("🟢 Raw Tick Preview (last 10)")
    if st.session_state.last_ticks:
        st.json(st.session_state.last_ticks)
    else:
        st.write("⚙️ Waiting for ticks...")
    
    if st.button("📥 Download Logs"):
        try:
            with open("live_trading.log", "r") as f:
                log_contents = f.read()
            st.download_button("Download log file", log_contents, "live_trading.log", "text/plain")
        except Exception as e:
            st.error(f"Log read error: {e}")
