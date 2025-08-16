import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from backtest import run_backtest_simulation  # Ensure backtest.py is accessible


# === Streamlit Configuration ===
st.set_page_config(layout="wide")
st.title("📈 Trading Signal Dashboard with ATR Position Sizing and Buy & Hold")


# === Sidebar: File Upload and Parameters ===
st.sidebar.header("Upload Files")
csv_file = st.sidebar.file_uploader("📂 Upload `5m_signals_enhanced_.csv`", type="csv")
optimization_file = st.sidebar.file_uploader("📂 Upload `grid_search_results_.csv`", type="csv")

st.sidebar.header("Position Sizing Parameters")
starting_capital = st.sidebar.number_input(
    "Starting Capital (₹)", min_value=10000, value=100000, step=5000
)
risk_per_trade_percent = st.sidebar.slider(
    "Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1
)
risk_per_trade = risk_per_trade_percent / 100  # decimal


# === Load Data and Run Backtest ===
optimization_results = None
df_signals, trades_df = None, None

if csv_file and optimization_file:
    if optimization_file.size == 0:
        st.error("The uploaded optimization file is empty. Please upload a valid CSV file.")
    else:
        df_signals = pd.read_csv(csv_file, parse_dates=["datetime"])
        df_signals.set_index("datetime", inplace=True)
        optimization_results = pd.read_csv(optimization_file)
        if optimization_results.empty:
            st.warning("The uploaded optimization file is empty. Please upload a valid file.")
        else:
            st.write("Optimization results loaded successfully!")
            st.write(optimization_results.head())

        trades = run_backtest_simulation(
            df_signals,
            trail_mult=2.0, time_limit=16, adx_target_mult=2.5,
            starting_capital=starting_capital, risk_per_trade=risk_per_trade
        )
        trades_df = pd.DataFrame(trades)


# === Tabs ===
tabs = st.tabs(["Signals", "Backtest", "Performance", "Optimization", "Insights"])

# ----- Tab 0: Signals -----
with tabs[0]:
    st.subheader("🔍 Enhanced Signals Data Exploration")
    if csv_file:
        min_date, max_date = df_signals.index.min(), df_signals.index.max()
        date_range = st.date_input("📅 Filter by Date Range", [min_date, max_date])
        filtered_signals = df_signals
        if len(date_range) == 2:
            filtered_signals = filtered_signals.loc[date_range[0] : date_range[1]]

        signal_option = st.selectbox("🎯 Filter by Signal Type", ["All", "Buy (1)", "Sell (-1)", "Neutral (0)"])
        if signal_option != "All":
            signal_map = {"Buy (1)": 1, "Sell (-1)": -1, "Neutral (0)": 0}
            filtered_signals = filtered_signals[filtered_signals["signal"] == signal_map[signal_option]]

        st.markdown("📊 Signal Summary")
        signal_counts = filtered_signals["signal"].value_counts().sort_index()
        st.write(signal_counts.rename({-1: "Sell (-1)", 0: "Neutral (0)", 1: "Buy (1)"}))

        st.markdown("⏱️ Signal Timeline")
        st.line_chart(filtered_signals["signal"])

        st.markdown("📄 Signal Table Preview")
        st.dataframe(filtered_signals[["predicted_label", "confidence", "signal", "position"]].head(100))
    else:
        st.info("Upload a signal CSV file to start.")


# ----- Tab 1: Backtest -----
with tabs[1]:
    st.subheader("📊 Backtest Trade Explorer")
    if csv_file and trades_df is not None and not trades_df.empty:
        trade_type = st.selectbox("Trade Type", options=["All", "Buy", "Short Sell"])
        min_dur, max_dur = int(trades_df["duration_min"].min()), int(trades_df["duration_min"].max())
        dur_range = st.slider("Trade Duration (minutes)", min_dur, max_dur, (min_dur, max_dur))

        filtered_trades = trades_df.copy()
        if trade_type != "All":
            filtered_trades = filtered_trades[filtered_trades["trade_type"] == trade_type]
        filtered_trades = filtered_trades[
            (filtered_trades["duration_min"] >= dur_range[0]) & (filtered_trades["duration_min"] <= dur_range[1])
        ]

        st.write(f"📦 Showing {len(filtered_trades)} trades")
        st.markdown("#### 📄 Filtered Trade Log")
        st.dataframe(filtered_trades.sort_values(by="exit_time", ascending=False).reset_index(drop=True))

        st.markdown("#### 🕒 Trade Duration Timeline")
        fig, ax = plt.subplots(figsize=(12, 4))
        for idx, row in filtered_trades.iterrows():
            ax.plot(
                [row["entry_time"], row["exit_time"]],
                [idx, idx],
                color="green" if row["net_pnl"] >= 0 else "red",
                linewidth=2,
                alpha=0.7,
            )
        ax.set_xlabel("Time")
        ax.set_title("Entry to Exit Duration of Each Trade")
        ax.grid(True)
        st.pyplot(fig)

        with st.expander("ℹ️ What does each column mean?"):
            st.markdown(
                """
            - `entry_time`: Trade start time
            - `exit_time`: Trade end time
            - `entry_price`: Entry price
            - `final_exit_price`: Exit price
            - `pnl_final`: Gross profit/loss
            - `net_pnl`: Profit after fees
            - `position_size`: Number of shares/contracts traded
            - `capital_after_trade`: Capital value after trade exit
            - `trade_type`: Buy or Short Sell
            - `duration_min`: Trade length (in minutes)
            """
            )
    else:
        st.info("Upload signal CSV and generate trades to view this tab.")


# ----- Tab 2: Performance -----
with tabs[2]:
    st.subheader("📈 Strategy Performance Summary")
        if csv_file and trades_df is not None and not trades_df.empty:
            total_trades = len(trades_df)
            profitable_trades = (trades_df["pnl"] > 0).sum()
            win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
            total_fees = trades_df["fees"].sum() if "fees" in trades_df.columns else 0
    
            # Calculate percentage returns relative to starting capital
            initial_capital = starting_capital  # from sidebar input
    
            gross_pnl = trades_df["pnl"].sum()
            net_pnl = trades_df["net_pnl"].sum()
            gross_return_pct = (gross_pnl / initial_capital) * 100
            net_return_pct = (net_pnl / initial_capital) * 100
    
            # Buy & Hold return percentage
            start_price = df_signals["close"].iloc[0]
            max_qty_buyhold = int(initial_capital // start_price)
            leftover_cash = initial_capital - max_qty_buyhold * start_price
            end_price = df_signals["close"].iloc[-1]
            buy_hold_pnl = (end_price - start_price) * max_qty_buyhold
            buy_hold_total_value = buy_hold_pnl + leftover_cash
            buy_hold_return_pct = ((buy_hold_total_value - initial_capital) / initial_capital) * 100
    
            # Split 6 metrics into two rows with 3 columns each
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)
    
            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{win_rate:.2f}%")
            col3.metric("Avg Duration", f"{trades_df['duration_min'].mean():.1f} min")
    
            col4.metric("Gross PnL", f"₹{gross_pnl:,.2f} ({gross_return_pct:.2f}%)")
            col5.metric("Net PnL", f"₹{net_pnl:,.2f} ({net_return_pct:.2f}%)")
            col6.metric("Total Fees", f"₹{total_fees:,.2f}")
    
            # Add an extra row for Buy & Hold return percentage
            st.markdown(f"**Buy & Hold Return:** {buy_hold_return_pct:.2f}%")
    
            # Plotting cumulative returns chart as before (remember to keep your existing plotting code here)
            # ...
        else:
            st.warning("No trades to display. Upload data and run backtest.")

        st.markdown("#### 📊 Cumulative Gross vs Net vs Buy & Hold")

        trades_df_sort = trades_df.sort_values("exit_time")
        trades_df_sort["cumulative_gross"] = trades_df_sort["pnl"].cumsum()
        trades_df_sort["cumulative_net"] = trades_df_sort["net_pnl"].cumsum()

        # Buy & Hold with max shares purchasable
        start_price = df_signals["close"].iloc[0]
        max_qty_buyhold = int(starting_capital // start_price)
        leftover_cash = starting_capital - max_qty_buyhold * start_price
        df_signals["buy_hold_return"] = (df_signals["close"] - start_price) * max_qty_buyhold

        aligned_df = pd.merge(
            trades_df_sort[["exit_time", "cumulative_gross", "cumulative_net"]],
            df_signals[["buy_hold_return"]],
            left_on="exit_time",
            right_index=True,
            how="inner",
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        aligned_df.set_index("exit_time")["buy_hold_return"].plot(
            ax=ax, label="Buy & Hold", linestyle="--", color="green"
        )
        aligned_df.set_index("exit_time")["cumulative_gross"].plot(ax=ax, label="Gross Strategy PnL", color="orange")
        aligned_df.set_index("exit_time")["cumulative_net"].plot(ax=ax, label="Net Strategy PnL", color="blue")
        
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        ax.set_title("Cumulative Returns: Strategy vs Buy & Hold")
        ax.set_ylabel("₹ Value")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        best_trade = trades_df_sort.loc[trades_df_sort["net_pnl"].idxmax()]
        worst_trade = trades_df_sort.loc[trades_df_sort["net_pnl"].idxmin()]

        colA, colB = st.columns(2)
        colA.success("**Best Trade**")
        colA.json(best_trade.to_dict())
        colB.error("**Worst Trade**")
        colB.json(worst_trade.to_dict())

        st.markdown("#### 🚰 Trade PnL Waterfall Chart")
        fig, ax = plt.subplots(figsize=(12, 5))
        cumulative = 0
        bottoms = []
        for pnl in trades_df_sort["net_pnl"]:
            bottoms.append(cumulative)
            cumulative += pnl
        ax.bar(
            range(len(trades_df_sort)),
            trades_df_sort["net_pnl"],
            bottom=bottoms,
            color=["green" if x >= 0 else "red" for x in trades_df_sort["net_pnl"]],
        )
        ax.set_title("Trade-by-Trade PnL Contribution")
        ax.set_xlabel("Trade Index")
        ax.set_ylabel("Net PnL")
        st.pyplot(fig)

        st.markdown("#### 🥧 Win vs Loss Breakdown")
        win_loss_counts = trades_df_sort["net_pnl"].apply(lambda x: "Win" if x > 0 else "Loss").value_counts()
        labels = win_loss_counts.index.tolist()
        sizes = win_loss_counts.values
        colors = ["#4CAF50", "#F44336"]
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=sizes,
                    textinfo="label+percent+value",
                    marker=dict(colors=colors),
                    hoverinfo="label+percent+value",
                    hole=0.3,
                    pull=[0.1 if label == "Win" else 0 for label in labels],
                )
            ]
        )
        fig.update_layout(title="Trade Outcome Distribution", title_x=0.5, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig)
    else:
        st.warning("No trades to display. Upload data and run backtest.")


# ----- Tab 3: Optimization -----
with tabs[3]:
    st.subheader("📊 Optimization Results")

    if optimization_results is None:
        st.info("Awaiting optimization file upload…")
    elif optimization_results.empty:
        st.warning("Uploaded optimization file is empty. Please upload a non-empty CSV.")
    else:
        df_opt = optimization_results.copy()
        threshold = st.slider("Select Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        filtered = df_opt[df_opt["ml_threshold"] >= threshold]
        st.markdown(f"**Showing {len(filtered)} rows with ml_threshold ≥ {threshold:.2f}:**")
        st.dataframe(filtered)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(filtered["win_rate"], filtered["total_pnl"], color="blue", alpha=0.6)
        ax1.set_xlabel("Win Rate (%)")
        ax1.set_ylabel("Total PnL")
        ax1.set_title("Win Rate vs Total PnL")
        ax1.grid(True)
        st.pyplot(fig1)

        if {"param1", "param2", "win_rate"}.issubset(filtered.columns):
            pivot = filtered.pivot(index="param2", columns="param1", values="win_rate")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            c = ax2.pcolormesh(pivot.columns.astype(float), pivot.index.astype(float), pivot.values, cmap="Blues", shading="auto")
            fig2.colorbar(c, ax=ax2)
            ax2.set_xlabel("Parameter 1")
            ax2.set_ylabel("Parameter 2")
            ax2.set_title("Parameter Grid Search – Win Rate Heatmap")
            st.pyplot(fig2)


# ----- Tab 4: Insights -----
with tabs[4]:
    st.subheader("📊 Insights")
    if csv_file and trades_df is not None and not trades_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(trades_df["duration_min"], bins=30, color="skyblue", edgecolor="black")
        ax.set_title("Distribution of Trade Duration (Minutes)")
        ax.set_xlabel("Duration (minutes)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        trades_df["cumulative_net"] = trades_df["net_pnl"].cumsum()
        start_price = df_signals["close"].iloc[0]
        df_signals["buy_hold_return"] = df_signals["close"] - start_price

        aligned_df = pd.merge(
            trades_df[["exit_time", "cumulative_net"]],
            df_signals[["buy_hold_return"]],
            left_on="exit_time",
            right_index=True,
            how="inner",
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        aligned_df.set_index("exit_time")["cumulative_net"].plot(ax=ax, label="Net Strategy PnL", color="blue")
        aligned_df.set_index("exit_time")["buy_hold_return"].plot(ax=ax, label="Buy & Hold", color="gray", linestyle="--")
        ax.set_title("Cumulative Performance: Strategy vs Buy & Hold")
        ax.set_ylabel("₹ Value")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        trades_df["win_rate"] = (trades_df["pnl"] > 0).rolling(window=10).mean() * 100
        fig, ax = plt.subplots(figsize=(12, 5))
        trades_df.set_index("exit_time")["win_rate"].plot(ax=ax, color="green")
        ax.set_title("Win Rate Over Time (10-period Rolling)")
        ax.set_ylabel("Win Rate (%)")
        ax.set_xlabel("Date")
        ax.grid(True)
        st.pyplot(fig)

        # Risk-adjusted metrics
        def calculate_sharpe_ratio(returns, risk_free_rate=0):
            return (returns.mean() - risk_free_rate) / returns.std()

        def calculate_sortino_ratio(returns, risk_free_rate=0):
            downside_returns = returns[returns < 0]
            return (returns.mean() - risk_free_rate) / downside_returns.std()

        def calculate_max_drawdown(cumulative_returns):
            drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
            return drawdowns.min()

        def calculate_volatility(returns):
            return returns.std()

        trades_df["daily_returns"] = trades_df["net_pnl"].pct_change().fillna(0)
        sharpe_ratio = calculate_sharpe_ratio(trades_df["daily_returns"])
        sortino_ratio = calculate_sortino_ratio(trades_df["daily_returns"])
        cum_net = trades_df["net_pnl"].cumsum()
        norm_cum = cum_net / cum_net.iloc[0]
        max_drawdown = calculate_max_drawdown(norm_cum)
        volatility = calculate_volatility(trades_df["daily_returns"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        col2.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
        col3.metric("Max Drawdown", f"{max_drawdown:.2%}")
        col4.metric("Volatility (σ)", f"{volatility:.2f}")

        total_net_pnl = trades_df["net_pnl"].sum()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter([sharpe_ratio], [total_net_pnl], color="blue", alpha=0.7, s=100)
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Total Net PnL")
        ax.set_title("Sharpe Ratio vs Total Net PnL")
        ax.grid(True)
        st.pyplot(fig)

    else:
        st.warning("No trades data available to display insights.")
