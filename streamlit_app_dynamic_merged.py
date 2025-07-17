import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import run_backtest_simulation  # Ensure this import matches the file location

# === Streamlit Configuration ===
st.set_page_config(layout="wide")
st.title("📈 Trading Signal Dashboard")

# === File Upload Section ===
st.sidebar.header("Upload Files")
csv_file = st.sidebar.file_uploader("📂 Upload `5m_signals_enhanced_<STOCK>.csv`", type="csv")
optimization_file = st.sidebar.file_uploader("📂 Upload `grid_search_results_BEL.csv`", type="csv")

# === File Handling and Backtest ===
if csv_file and optimization_file:
    # Load the Enhanced Signal Data
    df_signals = pd.read_csv(csv_file, parse_dates=['datetime'])
    df_signals.set_index('datetime', inplace=True)

    # Load the Optimization Results Data
    optimization_results = pd.read_csv(optimization_file)

    # === Check the column names of the optimization file to prevent KeyError ===
    st.write("### Optimization Results Columns:")
    st.write(optimization_results.columns)  # Display the column names

    # === Backtest Simulation ===
    trades = run_backtest_simulation(df_signals)
    trades_df = pd.DataFrame(trades)

    # === Performance Summary: Display Key Metrics ===
    total_trades = len(trades_df)
    profitable_trades = (trades_df['pnl'] > 0).sum()
    win_rate = (profitable_trades / total_trades) * 100
    avg_pnl = trades_df['pnl'].mean()
    total_fees = trades_df['fees'].sum() if 'fees' in trades_df.columns else 0

    # === Tabs Layout using `st.tabs()` ===
    tabs = st.tabs(["Signals", "Backtest", "Performance", "Optimization", "Duration Histogram"])

    with tabs[0]:
        st.subheader("### Enhanced Signals Data")
        st.write(df_signals[['predicted_label', 'confidence', 'signal', 'position']].head())

    with tabs[1]:
        if not trades_df.empty:
            st.subheader("### Backtest Results")
            st.write(f"Total Trades: {total_trades}")
            st.dataframe(trades_df)

    with tabs[2]:
        if not trades_df.empty:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{win_rate:.2f}%")
            col3.metric("Avg Duration", f"{trades_df['duration_min'].mean():.1f} min")
            col4.metric("Gross PnL", f"{trades_df['pnl'].sum():.2f}")
            col5.metric("Net PnL", f"{trades_df['net_pnl'].sum():.2f}")
            col6.metric("Total Fees", f"{total_fees:.2f}")

            st.subheader("📉 Cumulative PnL Over Time")
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            st.line_chart(trades_df.set_index('exit_time')['cumulative_pnl'])

    with tabs[3]:
        st.subheader("### Optimization Results")
        threshold_filter = st.slider("Select Confidence Threshold", 0.0, 1.0, 0.5)
        filtered_results = optimization_results[optimization_results['ml_threshold'] >= threshold_filter]
        st.write(filtered_results)

        st.subheader("📊 Win Rate vs Total PnL")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(optimization_results['win_rate'], optimization_results['total_pnL'], color='blue')
        ax.set_title("Win Rate vs Total PnL")
        ax.set_xlabel("Win Rate (%)")
        ax.set_ylabel("Total PnL")
        ax.grid(True)
        st.pyplot(fig)

    with tabs[4]:
        st.subheader("📊 Trade Duration Histogram")
        if not trades_df.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.hist(trades_df['duration_min'], bins=30, color='skyblue', edgecolor='black')
            ax2.set_title("Trade Duration (Minutes)")
            ax2.set_xlabel("Duration (minutes)")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)

else:
    st.warning("Please upload the necessary CSV files to proceed with the backtest.")
