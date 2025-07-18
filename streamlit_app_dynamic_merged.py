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
    tabs = st.tabs(["Signals", "backtest", "Performance", "Optimization", "Insights"])

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

        if not trades_df.empty:
            # === Filters ===
            st.markdown("#### 🔍 Filter Trades")
            trade_type = st.selectbox("Trade Type", options=["All", "Buy", "Short Sell"])
            min_dur, max_dur = int(trades_df['duration_min'].min()), int(trades_df['duration_min'].max())
            dur_range = st.slider("Trade Duration (minutes)", min_dur, max_dur, (min_dur, max_dur))
    
            filtered_trades = trades_df.copy()
            if trade_type != "All":
                filtered_trades = filtered_trades[filtered_trades['trade_type'] == trade_type]
            filtered_trades = filtered_trades[
                (filtered_trades['duration_min'] >= dur_range[0]) &
                (filtered_trades['duration_min'] <= dur_range[1])
            ]
    
            st.write(f"📦 Showing {len(filtered_trades)} trades")
    
            # === Trade Table ===
            st.markdown("#### 📄 Filtered Trade Log")
            st.dataframe(filtered_trades.sort_values(by='exit_time', ascending=False).reset_index(drop=True))
    
            # === Trade Timeline Plot ===
            st.markdown("#### 🕒 Trade Duration Timeline")
            fig, ax = plt.subplots(figsize=(12, 4))
            for idx, row in filtered_trades.iterrows():
                ax.plot([row['entry_time'], row['exit_time']], [idx, idx],
                        color='green' if row['net_pnl'] >= 0 else 'red',
                        linewidth=2, alpha=0.7)
            ax.set_xlabel("Time")
            ax.set_title("Entry to Exit Duration of Each Trade")
            ax.grid(True)
            st.pyplot(fig)
    
            # === Trade Column Definitions (Helpful Tooltip) ===
            with st.expander("ℹ️ What does each column mean?"):
                st.markdown("""
                - `entry_time`: Trade start time  
                - `exit_time`: Trade end time  
                - `entry_price`: Entry price  
                - `final_exit_price`: Exit price  
                - `pnl_final`: Gross profit/loss  
                - `net_pnl`: Profit after fees  
                - `trade_type`: Buy or Short  
                - `duration_min`: Trade length (in minutes)  
                """)
    
        else:
            st.info("No trades found in the uploaded file.")


    with tabs[2]:
        st.subheader("📈 Strategy Performance Summary")
    
        if not trades_df.empty:
            # === KPIs ===
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{win_rate:.2f}%")
            col3.metric("Avg Duration", f"{trades_df['duration_min'].mean():.1f} min")
            col4.metric("Gross PnL", f"{trades_df['pnl'].sum():.2f}")
            col5.metric("Net PnL", f"{trades_df['net_pnl'].sum():.2f}")
            col6.metric("Total Fees", f"{total_fees:.2f}")
    
            st.markdown("#### 📊 Cumulative Gross vs Net vs Buy & Hold")

            # Calculate cumulative strategy PnLs
            trades_df = trades_df.sort_values('exit_time')
            trades_df['cumulative_gross'] = trades_df['pnl'].cumsum()
            trades_df['cumulative_net'] = trades_df['net_pnl'].cumsum()
            
            # Calculate Buy & Hold return series
            start_price = df_signals['close'].iloc[0]
            df_signals['buy_hold_return'] = df_signals['close'] - start_price
            
            # Merge all into one timeline based on exit_time
            aligned_df = pd.merge(
                trades_df[['exit_time', 'cumulative_gross', 'cumulative_net']],
                df_signals[['buy_hold_return']],
                left_on='exit_time',
                right_index=True,
                how='inner'
            )
            
            # Plot the three curves
            fig, ax = plt.subplots(figsize=(12, 5))
            aligned_df.set_index('exit_time')['buy_hold_return'].plot(ax=ax, label="Buy & Hold", linestyle='--', color='gray')
            aligned_df.set_index('exit_time')['cumulative_gross'].plot(ax=ax, label="Gross Strategy PnL", color='orange')
            aligned_df.set_index('exit_time')['cumulative_net'].plot(ax=ax, label="Net Strategy PnL", color='blue')
            
            ax.set_title("Cumulative Returns: Strategy vs Buy & Hold")
            ax.set_ylabel("₹ Value")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
    
            # === Best & Worst Trades ===
            st.markdown("#### 🏆 Best and Worst Trades")
            colA, colB = st.columns(2)
            best_trade = trades_df.loc[trades_df['net_pnl'].idxmax()]
            worst_trade = trades_df.loc[trades_df['net_pnl'].idxmin()]
            colA.success("**Best Trade**")
            colA.json(best_trade.to_dict())
            colB.error("**Worst Trade**")
            colB.json(worst_trade.to_dict())
    
            # === PnL Waterfall Chart ===
            st.markdown("#### 🚰 Trade PnL Waterfall Chart")
            fig, ax = plt.subplots(figsize=(12, 5))
            cumulative = 0
            bottoms = []
            for pnl in trades_df['net_pnl']:
                bottoms.append(cumulative)
                cumulative += pnl
            ax.bar(range(len(trades_df)), trades_df['net_pnl'], bottom=bottoms,
                   color=['green' if x >= 0 else 'red' for x in trades_df['net_pnl']])
            ax.set_title("Trade-by-Trade PnL Contribution")
            ax.set_xlabel("Trade Index")
            ax.set_ylabel("Net PnL")
            st.pyplot(fig)
    
            # === Download Button ===
            st.markdown("#### 💾 Download Trade Data")
            csv_download = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Trades as CSV", csv_download, file_name="backtest_trades.csv", mime='text/csv')

            import plotly.graph_objects as go

            st.markdown("#### 🥧 Win vs Loss Breakdown")
            
            # Count wins and losses
            win_loss_counts = trades_df['net_pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss').value_counts()
            labels = win_loss_counts.index.tolist()
            sizes = win_loss_counts.values
            colors = ['#4CAF50', '#F44336']  # Green for wins, red for losses
            
            # Create Pie Chart with Plotly
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=sizes,
                textinfo='label+percent+value',  # Show label, percentage, and value on slices
                marker=dict(colors=colors),  # Custom colors
                hoverinfo='label+percent+value',  # Hover shows all info
                hole=0.3,  # Make it a donut chart (optional)
                pull=[0.1 if label == 'Win' else 0 for label in labels],  # Explode the 'Win' slice slightly
            )])
            
            fig.update_layout(
                title="Trade Outcome Distribution",
                title_x=0.5,
                template="plotly_dark",  # Optional: change chart theme
                showlegend=False
            )
            
            st.plotly_chart(fig)

    
        else:
            st.warning("No trades to display. Upload data to begin.")

    with tabs[3]:
        st.subheader("📊 Optimization Results")

    # File Upload Section (for optimization results)
    optimization_file = st.sidebar.file_uploader("📂 Upload `grid_search_results_BEL.csv`", type="csv")

    if optimization_file:
        # Load the optimization results CSV
        optimization_results = pd.read_csv(optimization_file)
        st.write("Optimization results loaded successfully!")
        st.write(optimization_results.head())  # Display the first few rows to check the content

        # Check the columns of the uploaded file
        st.write("Columns in the uploaded file:", optimization_results.columns)

        # Filter based on the confidence threshold slider
        threshold_filter = st.slider("Select Confidence Threshold", 0.0, 1.0, 0.5)
        filtered_results = optimization_results[optimization_results['ml_threshold'] >= threshold_filter]

        # Show filtered results or warn if no results match the threshold
        if filtered_results.empty:
            st.warning(f"No results found above the selected threshold {threshold_filter:.2f}")
        else:
            st.write(filtered_results)

            # === Scatter Plot: Win Rate vs Total PnL ===
            st.markdown("#### 📉 Win Rate vs Total PnL")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(filtered_results['win_rate'], filtered_results['total_pnl'], color='blue', alpha=0.5)
            ax.set_xlabel("Win Rate (%)")
            ax.set_ylabel("Total PnL")
            ax.set_title("Win Rate vs Total PnL")
            ax.grid(True)
            st.pyplot(fig)

            # === Parameter Combination Grid Visualization (if applicable) ===
            if 'param1' in filtered_results.columns and 'param2' in filtered_results.columns:
                st.markdown("#### 📊 Parameter Combination Grid")
                param1_values = filtered_results['param1'].unique()
                param2_values = filtered_results['param2'].unique()

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                c = ax2.pcolormesh(param1_values, param2_values, filtered_results.pivot('param2', 'param1', 'win_rate'), cmap='Blues')
                fig2.colorbar(c, ax=ax2)
                ax2.set_xlabel("Parameter 1")
                ax2.set_ylabel("Parameter 2")
                ax2.set_title("Parameter Grid Search")
                st.pyplot(fig2)

    else:
        optimization_results = pd.DataFrame()  # If no file is uploaded
        st.warning("No optimization results found. Please upload a valid CSV file.")

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
