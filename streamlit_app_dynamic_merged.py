import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import run_backtest_simulation  # Assuming backtest.py is in the same directory
import plotly.graph_objects as go

# === Streamlit Configuration ===
st.set_page_config(layout="wide")
st.title("📈 Trading Signal Dashboard")

# === File Upload Section ===
st.sidebar.header("Upload Files")
csv_file = st.sidebar.file_uploader("📂 Upload `5m_signals_enhanced_<STOCK>.csv`", type="csv")
optimization_file = st.sidebar.file_uploader("📂 Upload `grid_search_results_BEL.csv`", type="csv")

# === File Handling and Backtest ===
if csv_file and optimization_file:
    # Check if the optimization file is empty
    file_size = optimization_file.size
    if file_size == 0:
        st.error("The uploaded optimization file is empty. Please upload a valid CSV file.")
    else:
        try:
            # Load the Enhanced Signal Data
            df_signals = pd.read_csv(csv_file, parse_dates=['datetime'])
            df_signals.set_index('datetime', inplace=True)

            # Load the Optimization Results Data
            optimization_results = pd.read_csv(optimization_file)
            st.write("Optimization results loaded successfully!")
            st.write(optimization_results.head())  # Display the first few rows to check the content

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
            tabs = st.tabs(["Signals", "Backtest", "Performance", "Optimization", "Insights"])

            with tabs[0]:
                st.subheader("🔍 Enhanced Signals Data Exploration")
            
                # --- Filter Controls ---
                st.markdown("📅 Filter by Date Range")
                min_date, max_date = df_signals.index.min(), df_signals.index.max()
                date_range = st.date_input("Select range", [min_date, max_date])
                if len(date_range) == 2:
                    df_signals = df_signals.loc[date_range[0]:date_range[1]]
            
                st.markdown("🎯 Filter by Signal Type")
                signal_option = st.selectbox("Show Signals", ["All", "Buy (1)", "Sell (-1)", "Neutral (0)"])
                if signal_option != "All":
                    signal_map = {"Buy (1)": 1, "Sell (-1)": -1, "Neutral (0)": 0}
                    df_signals = df_signals[df_signals['signal'] == signal_map[signal_option]]
            
                # --- Signal Summary ---
                st.markdown("📊 Signal Summary")
                signal_counts = df_signals['signal'].value_counts().sort_index()
                st.write(signal_counts.rename({-1: "Sell (-1)", 0: "Neutral (0)", 1: "Buy (1)"}))
                        
                # --- Line Chart: Signal Over Time ---
                st.markdown("⏱️ Signal Timeline")
                st.line_chart(df_signals['signal'])
            
                # --- Data Preview ---
                st.markdown("📄 Signal Table Preview")
                st.dataframe(df_signals[['predicted_label', 'confidence', 'signal', 'position']].head(100))

            with tabs[1]:
                st.subheader("📊 Backtest Trade Explorer")
                # Code for the backtest tab...
                pass

            with tabs[2]:
                st.subheader("📈 Strategy Performance Summary")
                # Code for the performance tab...
                pass

            with tabs[3]:
                st.subheader("📊 Optimization Results")
                # Code for the optimization tab...
                pass

        except pd.errors.EmptyDataError:
            st.error("The uploaded optimization file is empty.")
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

else:
    st.warning("Please upload the necessary CSV files to proceed with the backtest.")
