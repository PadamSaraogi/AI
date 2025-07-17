import pandas as pd
import numpy as np

def run_backtest_simulation(df, trail_mult=2.0, time_limit=16, adx_target_mult=2.5):
    trades = []  # Store trade information
    in_trade = False  # Flag to check if we are in a trade
    cooldown = 0  # Cooldown period after each trade
    COOLDOWN_BARS = 0  # Set cooldown to 0 for testing to allow immediate re-entry
    STOP_MULT = 1.0  # Stop loss multiplier based on ATR (Average True Range)

    # Set the exit time (3:25 PM) and re-entry time (9:00 AM)
    EXIT_TIME = pd.to_datetime("15:25:00").time()  # 15:25 in 24-hour format
    REENTER_TIME = pd.to_datetime("09:00:00").time()  # 09:00 AM in 24-hour format

    # Initialize trade exit tracker
    last_trade_exit = None  # This will store the exit time of the last trade
    entry_price = None
    entry_sig = None
    stop_price = None
    tp_full = None
    trail_price = None
    entry_idx = None

    # Debugging: Check how many signals are non-zero
    print(f"Total Valid Signals: {len(df[df['signal'] != 0])}")  # Check the number of non-zero signals

    # Iterate through the signal dataframe
    for i in range(1, len(df)):
        # Extract relevant data from the current row (index i)
        sig = df['signal'].iat[i]
        price = df['close'].iat[i]
        atr = df['ATR'].iat[i]
        adx = df['ADX14'].iat[i]
        trade_time = df.index[i].time()  # Get the time part of the timestamp
        trade_date = df.index[i].date()

        # Debugging: Print signal data and time
        print(f"Signal: {sig}, Trade Date: {trade_date}, Trade Time: {trade_time}")

        # If there is an open trade from the previous day, re-enter at the start of the next day
        if last_trade_exit and trade_time >= REENTER_TIME and trade_date > last_trade_exit.date():
            # Ensure that the re-entry logic only happens once per day
            if in_trade:
                print(f"Re-entering trade on {trade_date} at {trade_time}, Last Exit: {last_trade_exit}")
                entry_price = price
                entry_sig = sig
                stop_price = entry_price - STOP_MULT * atr * entry_sig  # Stop loss
                tp_full = entry_price + adx_target_mult * atr * entry_sig  # Full target price
                trail_price = entry_price  # Initial trailing stop
                in_trade = True
                entry_idx = i  # Store the entry index
                continue

        # If not already in trade and there's a signal (Buy or Sell)
        if not in_trade and sig != 0:
            entry_price = price
            entry_sig = sig
            stop_price = entry_price - STOP_MULT * atr * entry_sig  # Stop loss
            tp_full = entry_price + adx_target_mult * atr * entry_sig  # Full target price
            trail_price = entry_price  # Initial trailing stop
            in_trade = True
            entry_idx = i  # Store the entry index
            print(f"Entering trade on {trade_date} at {trade_time}, Entry Price: {entry_price}")
            continue

        # If already in trade, manage the trade
        if in_trade:
            duration = i - entry_idx  # Calculate trade duration in bars (intraday)
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
                duration >= time_limit  # Limit to intraday (same day)
            )

            # Force exit at 3:25 PM (Exit Time)
            if trade_time >= EXIT_TIME:
                hit_exit = True  # Exit the trade at 3:25 PM
                print(f"Exit at 3:25 PM, Exit Price: {price_now}")

            # If exit condition is met, close the trade
            if hit_exit:
                final_exit_price = price_now
                pnl_full = final_exit_price - entry_price if entry_sig == 1 else entry_price - final_exit_price  # PnL for full exit

                total_pnl = pnl_full  # Since we're not tracking half-exit, total PnL is just full exit

                # Calculate intraday fees (this can be customized, depending on your fee model)
                fees = 0  # Set fees to 0 for now (calculate according to your fee structure)

                net_pnl = total_pnl - fees  # Calculate net profit/loss after fees

                # Store trade data
                trades.append({
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'final_exit_price': final_exit_price,
                    'pnl_final': pnl_full,
                    'fees': fees,
                    'net_pnl': net_pnl,  # Add net PnL
                    'pnl': total_pnl,
                    'trade_type': 'Buy' if entry_sig == 1 else 'Short Sell',
                    'duration_min': duration  # Add duration in minutes (intraday)
                })

                # Reset trade variables for next trade
                in_trade = False
                cooldown = COOLDOWN_BARS  # Set cooldown period
                last_trade_exit = df.index[i]  # Record the exit time of the current trade

    return trades
