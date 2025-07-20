import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from backtest import run_backtest_simulation  # your backtest logic

# === Fee Calculator for ICICI Direct Prime-299 Plan ===
def calculate_intraday_fees(entry_price, exit_price, quantity, plan_rate=0.00025):
    value_buy  = entry_price  * quantity
    value_sell = exit_price   * quantity

    # 1) Brokerage both legs
    brok_buy  = plan_rate * value_buy
    brok_sell = plan_rate * value_sell

    # 2) STT on sell leg only (0.025%)
    stt = 0.00025 * value_sell

    # 3) Turnover-based charges (both legs)
    turnover = value_buy + value_sell
    exch_fee = 0.0000297 * turnover    # ₹297 per ₹1 Cr
    sebi_fee = 0.000001  * turnover    # ₹10 per ₹1 Cr

    # 4) Stamp duty on buy leg only (₹300 per ₹1 Cr)
    stamp = 0.00003 * value_buy

    # 5) GST @18% on (brokerage + exch_fee + sebi_fee)
    gst = 0.18 * (brok_buy + brok_sell + exch_fee + sebi_fee)

    return brok_buy + brok_sell + stt + exch_fee + sebi_fee + stamp + gst

# === Risk-Adjusted Metric Functions ===
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    return (returns.mean() - risk_free_rate) / returns.std() if returns.std() else np.nan

def calculate_sortino_ratio(returns, risk_free_rate=0):
    downside = returns[returns < 0]
    return (returns.mean() - risk_free_rate) / downside.std() if downside.std() else np.nan

def calculate_max_drawdown(cum_returns):
    dd = cum_returns / cum_returns.cummax() - 1
    return dd.min()

def calculate_volatility(returns):
    return returns.std()

# === Streamlit Page Config ===
st.set_page_config(layout="wide")
st.title("📈 Trading Signal Dashboard")

# === Sidebar File Uploader ===
st.sidebar.header("Upload Files")
csv_file          = st.sidebar.file_uploader("Upload 5m_signals_enhanced_<STOCK>.csv", type="csv")
optimization_file = st.sidebar.file_uploader("Upload grid_search_results_<STOCK>.csv", type="csv")

optimization_results = None

if csv_file and optimization_file:
    if optimization_file.size == 0:
        st.error("Optimization file is empty. Please upload a valid CSV.")
    else:
        # --- Load Signals ---
        df_signals = pd.read_csv(csv_file, parse_dates=['datetime'])
        df_signals.set_index('datetime', inplace=True)

        # --- Load Optimization Results ---
        optimization_results = pd.read_csv(optimization_file)
        if optimization_results.empty:
            st.warning("Optimization CSV has no rows. Please upload a non-empty file.")
        else:
            # --- Run Backtest Simulation ---
            trades = run_backtest_simulation(df_signals)
            trades_df = pd.DataFrame(trades)

            # --- Calculate Intraday Fees & Net PnL ---
            trades_df['fees']    = trades_df.apply(
                lambda r: calculate_intraday_fees(
                    r['entry_price'], r['exit_price'], r['quantity']
                ), axis=1
            )
            trades_df['net_pnl'] = trades_df['pnl'] - trades_df['fees']

            # --- Precompute Some Summary Metrics ---
            total_trades      = len(trades_df)
            total_fees        = trades_df['fees'].sum() if 'fees' in trades_df.columns else 0.0

            # --- Layout Tabs ---
            tabs = st.tabs(["Signals", "Backtest", "Performance", "Optimization", "Insights"])

            # === Tab 0: Signals ===
            with tabs[0]:
                st.subheader("🔍 Enhanced Signals Data Exploration")
                # Date filter
                min_date, max_date = df_signals.index.min(), df_signals.index.max()
                date_range = st.date_input("Select Date Range", [min_date, max_date])
                if len(date_range) == 2:
                    df_signals = df_signals.loc[date_range[0]:date_range[1]]

                # Signal type filter
                signal_option = st.selectbox("Show Signals", ["All", "Buy (1)", "Sell (-1)", "Neutral (0)"])
                if signal_option != "All":
                    mapping = {"Buy (1)":1, "Sell (-1)":-1, "Neutral (0)":0}
                    df_signals = df_signals[df_signals['signal'] == mapping[signal_option]]

                # Summary & charts
                st.write(df_signals['signal'].value_counts().sort_index())
                st.line_chart(df_signals['signal'])
                st.dataframe(df_signals[['predicted_label','confidence','signal','position']].head(100))

            # === Tab 1: Backtest Explorer ===
            with tabs[1]:
                st.subheader("📊 Backtest Trade Explorer")
                if not trades_df.empty:
                    # Trade filters
                    trade_type = st.selectbox("Trade Type", ["All","Buy","Short Sell"])
                    min_dur, max_dur = int(trades_df['duration_min'].min()), int(trades_df['duration_min'].max())
                    dur_range = st.slider("Trade Duration (min)", min_dur, max_dur, (min_dur, max_dur))

                    ft = trades_df.copy()
                    if trade_type != "All":
                        ft = ft[ft['trade_type']==trade_type]
                    ft = ft[(ft['duration_min']>=dur_range[0])&(ft['duration_min']<=dur_range[1])]

                    st.write(f"Showing {len(ft)} trades")
                    st.dataframe(ft.sort_values('exit_time', ascending=False).reset_index(drop=True))

                    # Duration timeline
                    fig, ax = plt.subplots(figsize=(12,4))
                    for i,row in ft.iterrows():
                        ax.plot([row['entry_time'], row['exit_time']], [i,i],
                                color='green' if row['net_pnl']>=0 else 'red', alpha=0.7)
                    ax.set_xlabel("Time"); ax.set_title("Entry→Exit Duration")
                    st.pyplot(fig)
                else:
                    st.info("No trades available.")

            # === Tab 2: Performance Summary ===
            with tabs[2]:
                st.subheader("📈 Strategy Performance Summary")
                if total_trades:
                    profitable = (trades_df['net_pnl']>0).sum()
                    win_rate   = profitable/total_trades*100
                    avg_dur    = trades_df['duration_min'].mean()
                    gross_pnl  = trades_df['pnl'].sum()
                    net_pnl    = trades_df['net_pnl'].sum()

                    cols = st.columns(6)
                    cols[0].metric("Total Trades", total_trades)
                    cols[1].metric("Win Rate", f"{win_rate:.2f}%")
                    cols[2].metric("Avg Duration", f"{avg_dur:.1f} min")
                    cols[3].metric("Gross PnL", f"{gross_pnl:.2f}")
                    cols[4].metric("Net PnL", f"{net_pnl:.2f}")
                    cols[5].metric("Total Fees", f"{total_fees:.2f}")

                    # Cumulative Returns vs Buy & Hold
                    trades_sorted = trades_df.sort_values('exit_time')
                    trades_sorted['cum_gross'] = trades_sorted['pnl'].cumsum()
                    trades_sorted['cum_net']   = trades_sorted['net_pnl'].cumsum()
                    start_price = df_signals['close'].iloc[0]
                    df_signals['bh_return'] = df_signals['close'] - start_price

                    aligned = pd.merge(
                        trades_sorted[['exit_time','cum_gross','cum_net']],
                        df_signals[['bh_return']],
                        left_on='exit_time', right_index=True, how='inner'
                    )

                    fig, ax = plt.subplots(figsize=(12,5))
                    aligned.set_index('exit_time')['bh_return'].plot(ax=ax, label="Buy & Hold", linestyle='--',color='gray')
                    aligned.set_index('exit_time')['cum_gross'].plot(ax=ax, label="Gross PnL",color='orange')
                    aligned.set_index('exit_time')['cum_net'].plot(ax=ax, label="Net PnL",color='blue')
                    ax.set_title("Cumulative Returns: Strategy vs Buy & Hold")
                    ax.set_xlabel("Date"); ax.set_ylabel("₹ Value"); ax.legend(); ax.grid(True)
                    st.pyplot(fig)

                else:
                    st.warning("No trades to summarize.")

            # === Tab 3: Optimization Results ===
            with tabs[3]:
                st.subheader("📊 Optimization Results")
                if optimization_results is None:
                    st.info("Upload optimization CSV to begin.")
                elif optimization_results.empty:
                    st.warning("Optimization file is empty.")
                else:
                    df_opt = optimization_results.copy()
                    threshold = st.slider("Confidence Threshold", 0.0,1.0,0.5,0.01)
                    filtered = df_opt[df_opt['ml_threshold']>=threshold]

                    st.markdown(f"Showing {len(filtered)} rows ≥ {threshold:.2f}")
                    st.dataframe(filtered)

                    # Win Rate vs Total PnL
                    fig1, ax1 = plt.subplots(figsize=(10,6))
                    ax1.scatter(filtered['win_rate'], filtered['total_pnl'], alpha=0.6)
                    ax1.set(xlabel="Win Rate (%)",ylabel="Total PnL",title="Win Rate vs Total PnL"); ax1.grid(True)
                    st.pyplot(fig1)

                    # Parameter grid heatmap
                    if {'param1','param2','win_rate'}.issubset(filtered.columns):
                        pivot = filtered.pivot('param2','param1','win_rate')
                        fig2, ax2 = plt.subplots(figsize=(10,6))
                        c = ax2.pcolormesh(pivot.columns,pivot.index,pivot.values,cmap='Blues',shading='auto')
                        fig2.colorbar(c,ax=ax2)
                        ax2.set(xlabel="Param1",ylabel="Param2",title="Win Rate Heatmap")
                        st.pyplot(fig2)

            # === Tab 4: Insights ===
            with tabs[4]:
                st.subheader("📊 Insights")

                # Trade Duration Histogram
                if not trades_df.empty:
                    fig, ax = plt.subplots(figsize=(10,6))
                    ax.hist(trades_df['duration_min'], bins=30, color='skyblue',edgecolor='black')
                    ax.set(title="Trade Duration Distribution", xlabel="Duration (min)", ylabel="Frequency")
                    st.pyplot(fig)
                else:
                    st.warning("No trades to plot duration histogram.")

                # Win Rate Over Time
                if not trades_df.empty:
                    trades_df['win_rate'] = (trades_df['pnl']>0).rolling(10).mean()*100
                    fig, ax = plt.subplots(figsize=(12,5))
                    trades_df.set_index('exit_time')['win_rate'].plot(ax=ax, color='green')
                    ax.set(title="10-period Rolling Win Rate",xlabel="Date",ylabel="Win Rate (%)"); ax.grid(True)
                    st.pyplot(fig)
                else:
                    st.warning("No trades to compute rolling win rate.")

                # Risk-Adjusted Metrics
                if not trades_df.empty:
                    # daily returns
                    trades_df['daily_returns'] = trades_df['net_pnl'].pct_change().fillna(0)
                    sharpe   = calculate_sharpe_ratio(trades_df['daily_returns'])
                    sortino  = calculate_sortino_ratio(trades_df['daily_returns'])
                    cum_net  = trades_df['net_pnl'].cumsum()
                    max_dd   = calculate_max_drawdown(cum_net / cum_net.iloc[0])
                    vol      = calculate_volatility(trades_df['daily_returns'])

                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    c2.metric("Sortino Ratio", f"{sortino:.2f}")
                    c3.metric("Max Drawdown", f"{max_dd:.2%}")
                    c4.metric("Volatility", f"{vol:.2f}")
                else:
                    st.warning("No trades for risk metrics.")

                # Sharpe vs Total Net PnL & Fees
                if not trades_df.empty:
                    total_net = trades_df['net_pnl'].sum()
                    total_fees= trades_df['fees'].sum()
                    s1,s2,s3 = st.columns(3)
                    s1.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    s2.metric("Total Net PnL", f"{total_net:.2f}")
                    s3.metric("Total Fees Paid", f"{total_fees:.2f}")

                    fig, ax = plt.subplots(figsize=(10,6))
                    ax.scatter([sharpe],[total_net], s=120, color='blue', alpha=0.7)
                    ax.set(xlabel="Sharpe Ratio", ylabel="Total Net PnL", title="Sharpe vs Net PnL")
                    ax.grid(True)
                    st.pyplot(fig)
                else:
                    st.warning("No trades to plot Sharpe vs Net PnL.")
