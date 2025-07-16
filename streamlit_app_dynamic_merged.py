import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import run_backtest_simulation  # Assuming backtest.py is in the same directory

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

    # Display a preview of the loaded files
    st.write("### Enhanced Signals Data")
    st.write(df_signals[['predicted_label', 'confidence', 'signal', 'position']].head())

    st.write("### Optimization Results")
    st.write(optimization_results)

    # === Backtest Simulation ===
    st.subheader("🔄 Backtest Simulation")
    trades = run_backtest_simulation(df_signals)
    trades_df = pd.DataFrame(trades)

    # Show Backtest Results if trades are available
    if not trades_df.empty:
        st.write(f"Total Trades: {len(trades_df)}")
        st.write("### Trade Details")
        st.dataframe(trades_df)

    # === Performance Summary: Display Key Metrics ===
    if not trades_df.empty:
        total_trades = len(trades_df)
        profitable_trades = (trades_df['pnl'] > 0).sum()
        win_rate = (profitable_trades / total_trades) * 100
        avg_pnl = trades_df['pnl'].mean()
        total_fees = trades_df['fees'].sum() if 'fees' in trades_df.columns else 0

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.2f}%")
        col3.metric("Avg Duration", f"{trades_df['duration_min'].mean():.1f} min")
        col4.metric("Gross PnL", f"{trades_df['pnl'].sum():.2f}")
        col5.metric("Net PnL", f"{trades_df['net_pnl'].sum():.2f}")
        col6.metric("Total Fees", f"{total_fees:.2f}")

    # === Cumulative PnL Chart ===
    if not trades_df.empty:
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        st.subheader("📉 Cumulative PnL Over Time")
        st.line_chart(trades_df.set_index('exit_time')['cumulative_pnl'])

    # === Optimization Results Analysis ===
    st.subheader("📊 Optimization Results")

    # Allow user to filter by confidence threshold
    threshold_filter = st.slider("Select Confidence Threshold", 0.0, 1.0, 0.5)
    filtered_results = optimization_results[optimization_results['ml_threshold'] >= threshold_filter]

    st.write(filtered_results)

    # === Visualize Optimization Results ===
    st.subheader("📊 Win Rate vs Total PnL")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(optimization_results['win_rate'], optimization_results['total_pnl'], color='blue')
    ax.set_title("Win Rate vs Total PnL")
    ax.set_xlabel("Win Rate (%)")
    ax.set_ylabel("Total PnL")
    ax.grid(True)
    st.pyplot(fig)

    # === Trade Duration Histogram ===
    if not trades_df.empty:
        st.subheader("📊 Trade Duration Histogram")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(trades_df['duration_min'], bins=30, color='skyblue', edgecolor='black')
        ax2.set_title("Trade Duration (Minutes)")
        ax2.set_xlabel("Duration (minutes)")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    # === Signal Overlay Heatmap ===
    st.subheader("📊 Signal Overlay Heatmap")
    if 'signal' in df_signals.columns:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        heatmap_data = df_signals[['signal', 'close']].copy()
        heatmap_data['signal'] = heatmap_data['signal'].map({1: 'Buy', -1: 'Sell', 0: 'Hold'})
        heatmap_data['time'] = heatmap_data.index
        ax3.scatter(heatmap_data['time'], heatmap_data['close'], c=heatmap_data['signal'].map({'Buy': 'green', 'Sell': 'red', 'Hold': 'gray'}))
        ax3.set_title("Signal Overlay Heatmap")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        st.pyplot(fig3)

# Display an error message if no files are uploaded
else:
    st.warning("Please upload the necessary CSV files to proceed with the backtest.")
