import pandas as pd
import numpy as np
import streamlit as st

def run_backtest_simulation(df, trail_mult=2.0, time_limit=16, adx_target_mult=2.5):
    trades = []  # Store trade information
    in_trade = False  # Flag to check if we are in a trade
    current_trade = None  # To store data for the ongoing trade

    # We will track the current date to ensure the trade closes within the same day or carries over
    current_date = None

    # Iterate through the signal dataframe
    for i in range(1, len(df)):
        # Extract relevant data from the current row (index i)
        sig = df['signal'].iat[i]
        price = df['close'].iat[i]
        atr = df['ATR'].iat[i]
        adx = df['ADX14'].iat[i]
        trade_date = df.index[i].date()
        trade_time = df.index[i].time()

        # If we are not in a trade and there's a signal (Buy or Sell)
        if not in_trade and sig != 0:
            if trade_date != current_date:
                current_date = trade_date  # Update the current trade date

            # Open new trade
            current_trade = {
                'entry_time': df.index[i],
                'entry_price': price,
                'entry_sig': sig,
                'stop_price': price - STOP_MULT * atr * sig,  # Stop loss
                'tp_full': price + adx_target_mult * atr * sig,  # Full target price
                'trail_price': price,  # Initial trailing stop
                'entry_idx': i,  # Store the entry index
                'in_trade': True
            }
            in_trade = True
            continue

        # If we are in a trade, manage the trade
        if in_trade:
            duration = i - current_trade['entry_idx']  # Calculate trade duration in bars (this is intraday)
            price_now = price
            atr_now = atr

            # Handle trailing stop for both Buy and Short positions
            if current_trade['entry_sig'] > 0:  # Long position (Buy)
                current_trade['trail_price'] = max(current_trade['trail_price'], price_now)
                trailing_stop = current_trade['trail_price'] - trail_mult * atr_now
            else:  # Short position (Sell)
                current_trade['trail_price'] = min(current_trade['trail_price'], price_now)
                trailing_stop = current_trade['trail_price'] + trail_mult * atr_now

            # Exit conditions (stop loss, target, trailing stop, or time limit)
            hit_exit = (
                (current_trade['entry_sig'] > 0 and (price_now <= current_trade['stop_price'] or price_now >= current_trade['tp_full'] or price_now <= trailing_stop)) or
                (current_trade['entry_sig'] < 0 and (price_now >= current_trade['stop_price'] or price_now <= current_trade['tp_full'] or price_now >= trailing_stop))
            )

            # Force exit at 3:20 PM (market close)
            if trade_time >= pd.to_datetime("15:20:00").time():
                hit_exit = True  # Exit the trade if the time is 3:20 PM or later

            # If exit condition is met, close the trade
            if hit_exit:
                final_exit_price = price_now
                pnl_full = final_exit_price - current_trade['entry_price'] if current_trade['entry_sig'] == 1 else current_trade['entry_price'] - final_exit_price  # PnL for full exit

                total_pnl = pnl_full  # Since we're not tracking half-exit, total PnL is just full exit

                # Calculate intraday fees (this can be customized, depending on your fee model)
                fees = 0  # Set fees to 0 for now (calculate according to your fee structure)

                net_pnl = total_pnl - fees  # Calculate net profit/loss after fees

                # Store trade data
                trades.append({
                    'entry_time': current_trade['entry_time'],
                    'exit_time': df.index[i],
                    'entry_price': current_trade['entry_price'],
                    'final_exit_price': final_exit_price,
                    'pnl_final': pnl_full,
                    'fees': fees,
                    'net_pnl': net_pnl,  # Add net PnL
                    'pnl': total_pnl,
                    'trade_type': 'Buy' if current_trade['entry_sig'] == 1 else 'Short Sell',
                    'duration_min': (df.index[i] - current_trade['entry_time']).total_seconds() / 60  # Actual duration in minutes
                })

                # Reset for the next day to carry over the trade
                in_trade = False
                current_trade = None

            # Carry over trade to next day if not closed by 3:20 PM
            if trade_time < pd.to_datetime("15:20:00").time():
                continue  # Carry over the open trade to the next day without closing it

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

    # === Trade Duration Histogram ===
    if not trades_df.empty:
        st.subheader("📊 Trade Duration Histogram")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(trades_df['duration_min'], bins=30, color='skyblue', edgecolor='black')
        ax2.set_title("Trade Duration (Minutes)")
        ax2.set_xlabel("Duration (minutes)")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
else:
    st.warning("Please upload the necessary CSV files to proceed with the backtest.")
