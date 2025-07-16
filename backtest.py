import pandas as pd
import numpy as np

def run_backtest_simulation(df, trail_mult=2.0, time_limit=16, adx_target_mult=2.5):
    trades = []  # Store trade information
    in_trade = False  # Flag to check if we are in a trade
    cooldown = 0  # Cooldown period after each trade
    COOLDOWN_BARS = 2  # Number of bars to wait before entering a new trade
    STOP_MULT = 1.0  # Stop loss multiplier based on ATR (Average True Range)

    # Iterate through the signal dataframe
    for i in range(1, len(df)):
        # Skip if we're in cooldown
        if cooldown > 0:
            cooldown -= 1
            continue

        # Extract relevant data from the current row (index i)
        sig = df['signal'].iat[i]
        price = df['close'].iat[i]
        atr = df['ATR'].iat[i]
        adx = df['ADX14'].iat[i]

        # If not already in trade and there's a signal (Buy or Sell)
        if not in_trade and sig != 0:
            entry_price = price
            entry_sig = sig
            stop_price = entry_price - STOP_MULT * atr * entry_sig  # Stop loss
            tp_full = entry_price + adx_target_mult * atr * entry_sig  # Full target price
            trail_price = entry_price  # Initial trailing stop
            in_trade = True
            entry_idx = i  # Store the entry index
            pnl_full = 0.0  # Initialize profit/loss variable for full exit
            continue

        # If already in trade, manage the trade
        if in_trade:
            duration = i - entry_idx  # Calculate trade duration in bars
            price_now = price
            atr_now = atr

            # Handle trailing stop for both Buy and Short positions
            if entry_sig > 0:  # Long position (Buy)
                trail_price = max(trail_price, price_now)
                trailing_stop = trail_price - trail_mult * atr_now
            else:  # Short position (Sell)
                trail_price = min(trail_price, price_now)
                trailing_stop = trail_price + trail_mult * atr_now

            # Exit conditions (stop loss, target, trailing stop, or time limit)
            hit_exit = (
                (entry_sig > 0 and (price_now <= stop_price or price_now >= tp_full or price_now <= trailing_stop)) or
                (entry_sig < 0 and (price_now >= stop_price or price_now <= tp_full or price_now >= trailing_stop)) or
                duration >= time_limit
            )

            # If exit condition is met, close the trade
            if hit_exit:
                final_exit_price = price_now
                pnl_full = final_exit_price - entry_price if entry_sig == 1 else entry_price - final_exit_price  # PnL for full exit

                total_pnl = pnl_full  # Since we're not tracking half-exit, total PnL is just full exit

                # Store trade data
                trades.append({
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'final_exit_price': final_exit_price,
                    'pnl_final': pnl_full,
                    'pnl': total_pnl,
                    'trade_type': 'Buy' if entry_sig == 1 else 'Short Sell'
                })

                # Reset trade variables for next trade
                in_trade = False
                cooldown = COOLDOWN_BARS  # Set cooldown period

    return trades

# === Streamlit: Correct File Upload Handling ===
import streamlit as st

# File Upload Section
csv_file = st.file_uploader("📂 Upload `5m_signals_enhanced_<STOCK>.csv`", type="csv")
if csv_file:
    # Read the uploaded file (csv_file is a BytesIO object)
    df = pd.read_csv(csv_file, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)

    st.write(f"### Enhanced Signals Data (First 5 rows):")
    st.write(df.head())  # Display the first 5 rows for preview

    # Run backtest using the enhanced signal data
    trades = run_backtest_simulation(df)

    # Convert trades into DataFrame for analysis
    trades_df = pd.DataFrame(trades)

    # Show Backtest Results
    if not trades_df.empty:
        st.write(f"Total Trades: {len(trades_df)}")
        st.write("### Trade Details")
        st.dataframe(trades_df)
