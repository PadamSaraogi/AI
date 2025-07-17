import pandas as pd
import numpy as np

def run_backtest_simulation(df, trail_mult=2.0, time_limit=16, adx_target_mult=2.5):
    trades = []  # Store trade information
    in_trade = False  # Flag to check if we are in a trade
    cooldown = 0  # Cooldown period after each trade
    COOLDOWN_BARS = 2  # Number of bars to wait before entering a new trade
    STOP_MULT = 1.0  # Stop loss multiplier based on ATR (Average True Range)

    # We will track the current date to ensure the trade closes within the same day
    current_date = None

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
        trade_date = df.index[i].date()

        # If not already in trade and there's a signal (Buy or Sell)
        if not in_trade and sig != 0:
            # Ensure that trades are initiated at the beginning of the day
            if trade_date != current_date:
                current_date = trade_date  # Update the current trade date

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
            duration = i - entry_idx  # Calculate trade duration in bars (this is intraday)
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

            # Force exit at the end of the day (market close)
            if trade_date != current_date:
                hit_exit = True  # Exit the trade if the date has changed (next day)

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
                    'duration_min': duration  # Add the duration in minutes (intraday)
                })

                # Reset trade variables for next trade
                in_trade = False
                cooldown = COOLDOWN_BARS  # Set cooldown period

    return trades
