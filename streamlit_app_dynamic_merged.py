import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from backtest import run_backtest_simulation

st.set_page_config(layout="wide")
st.markdown("""
    <div style='display: flex; align-items: center;'>
      <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/512px-React-icon.svg.png' width='60'>
      <h1 style='margin-left: 18px;'>Multi-Stock Trading Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

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
                    increase_percent = ((total_net_pnl - total_portfolio_capital) / total_portfolio_capital) * 100
                else:
                    increase_percent = 0.0
            else:
                max_drawdown = sharpe = sortino = volatility = np.nan

            st.markdown("### Portfolio Key Metrics")
            r1c1, r1c2, r1c3 = st.columns(3)
            r2c1, r2c2, r2c3 = st.columns(3)
            
            r1c1.metric("Total Trades", f"{total_trades}")
            r1c2.metric("Portfolio Value (₹)", f"₹{total_portfolio_capital:,.2f}")
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
            st.subheader("Portfolio Leaderboard")
            st.dataframe(df_summary.sort_values("Net PnL", ascending=False))

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


        st.subheader(f"{symbol_select.upper()} Drawdown")
        eq_cumret = equity_curve / equity_curve.iloc[0]
        drawdowns = eq_cumret / eq_cumret.cummax() - 1
        fig_dd, ax_dd = plt.subplots(figsize=(10, 4))
        drawdowns.plot(ax=ax_dd, color="red")
        ax_dd.set_ylabel("Drawdown")
        ax_dd.set_xlabel("Date")
        ax_dd.grid(True)
        st.pyplot(fig_dd)

        st.subheader(f"Candlestick Chart with Trades ({symbol_select.upper()})")
        signals_df = stock_data[symbol_select]['signals']
        if {'open','high','low','close'}.issubset(signals_df.columns):
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
            fig_candle.update_layout(title=f"{symbol_select.upper()} Price & Trades", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_candle)

        st.subheader(f"Trade Waterfall Chart ({symbol_select.upper()})")
        cum = 0
        bottoms = []
        for pnl in trades_df["net_pnl"]:
            bottoms.append(cum)
            cum += pnl
        fig_water, ax_water = plt.subplots(figsize=(12, 4))
        ax_water.bar(
            range(len(trades_df)),
            trades_df["net_pnl"],
            bottom=bottoms,
            color=["green" if x >= 0 else "red" for x in trades_df["net_pnl"]],
        )
        ax_water.set_xlabel("Trade Index")
        ax_water.set_ylabel("Net PnL (₹)")
        ax_water.set_title(f"Trade-by-Trade Net PnL Contribution ({symbol_select.upper()})")
        st.pyplot(fig_water)

        if not trades_df.empty:
            st.subheader(f"All Trades for {symbol_select.upper()}")
            st.dataframe(trades_df.sort_values("exit_time").reset_index(drop=True))
            csv_download = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                f"Download {symbol_select.upper()} Trades as CSV",
                csv_download,
                file_name=f"{symbol_select}_trades.csv",
                mime="text/csv"
            )
        else:
            st.info("No trade data available for selected symbol.")

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
                line=dict(width=4, dash='dash', color='black'),
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

        # --- 6. New: Per-stock Performance Summary ---
        perf_summary = []
        for symbol, eq_curve in all_equity_curves.items():
            total_return_pct = (eq_curve.iloc[-1] / eq_curve.iloc[0] - 1) * 100 if len(eq_curve) > 1 else 0
            drawdown_pct = ((eq_curve / eq_curve.cummax()) - 1).min() * 100 if len(eq_curve) > 1 else 0
            daily_rets = eq_curve.pct_change().dropna()
            sharpe_ratio = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else np.nan
            perf_summary.append({
                "Symbol": symbol.upper(),
                "Total Return (%)": total_return_pct,
                "Max Drawdown (%)": drawdown_pct,
                "Sharpe Ratio": sharpe_ratio,
            })
        df_perf = pd.DataFrame(perf_summary)
        for col in ["Total Return (%)", "Max Drawdown (%)", "Sharpe Ratio"]:
            df_perf[col] = df_perf[col].astype(float).map("{:.2f}".format)
        st.markdown("### Performance Summary")
        st.dataframe(df_perf)

        # --- 7. New: Download Combined Equity Curves CSV ---
        combined_eq_df = pd.concat(all_equity_curves.values(), axis=1)
        combined_eq_df.columns = [s.upper() for s in all_equity_curves.keys()]
        combined_eq_df.dropna(how='all', inplace=True)
        csv_data = combined_eq_df.to_csv().encode('utf-8')
        st.download_button(
            "Download Combined Equity Curves (CSV)",
            csv_data,
            file_name="combined_equity_curves.csv",
            mime="text/csv"
        )

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
        
        
            
