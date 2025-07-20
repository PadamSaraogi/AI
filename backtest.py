import pandas as pd
import numpy as np

def calculate_intraday_fees(entry_price, exit_price, quantity=1):
    turnover = (entry_price + exit_price) * quantity
    sell_turnover = exit_price * quantity
    buy_turnover = entry_price * quantity

    brokerage = min(0.00025 * turnover, 20)
    stt = 0.00025 * sell_turnover
    exchange_txn = 0.0000345 * turnover
    sebi_charges = 0.000001 * turnover
    stamp_duty = 0.00003 * buy_turnover
    gst = 0.18 * (brokerage + exchange_txn)

    total_fees = brokerage + stt + exchange_txn + sebi_charges + stamp_duty + gst
    return round(total_fees, 2)

def run_backtest_simulation(df, trail_mult=2.0, time_limit=16, adx_target_mult=2.5):
    trades = []
    in_trade = False
    cooldown = 0
    COOLDOWN_BARS = 0
    STOP_MULT = 1.0

    EXIT_TIME = pd.to_datetime("15:25:00").time()
    REENTER_TIME = pd.to_datetime("09:00:00").time()

    last_trade_exit = None
    entry_price = None
    entry_sig = None
    stop_price = None
    tp_full = None
    trail_price = None
    entry_idx = None

    print(f"Total Valid Signals: {len(df[df['signal'] != 0])}")

    for i in range(1, len(df)):
        sig = df['signal'].iat[i]
        price = df['close'].iat[i]
        atr = df['ATR'].iat[i]
        adx = df['ADX14'].iat[i]
        trade_time = df.index[i].time()
        trade_date = df.index[i].date()

        print(f"Signal: {sig}, Trade Date: {trade_date}, Trade Time: {trade_time}")

        if last_trade_exit and trade_time >= REENTER_TIME and trade_date > last_trade_exit.date():
            if not in_trade and sig != 0:
                print(f"Re-entering trade on {trade_date} at {trade_time}, Last Exit: {last_trade_exit}")
                entry_price = price
                entry_sig = sig
                stop_price = entry_price - STOP_MULT * atr * entry_sig
                tp_full = entry_price + adx_target_mult * atr * entry_sig
                trail_price = entry_price
                in_trade = True
                entry_idx = i
                continue

        if not in_trade and sig != 0:
            entry_price = price
            entry_sig = sig
            stop_price = entry_price - STOP_MULT * atr * entry_sig
            tp_full = entry_price + adx_target_mult * atr * entry_sig
            trail_price = entry_price
            in_trade = True
            entry_idx = i
            print(f"Entering trade on {trade_date} at {trade_time}, Entry Price: {entry_price}")
            continue

        if in_trade:
            duration = i - entry_idx
            price_now = price
            atr_now = atr

            if entry_sig > 0:
                trail_price = max(trail_price, price_now)
                trailing_stop = trail_price - trail_mult * atr_now
            else:
                trail_price = min(trail_price, price_now)
                trailing_stop = trail_price + trail_mult * atr_now

            hit_exit = (
                (entry_sig > 0 and (price_now <= stop_price or price_now >= tp_full or price_now <= trailing_stop)) or
                (entry_sig < 0 and (price_now >= stop_price or price_now <= tp_full or price_now >= trailing_stop)) or
                duration >= time_limit
            )

            if trade_time >= EXIT_TIME:
                hit_exit = True
                print(f"Exit at 3:25 PM, Exit Price: {price_now}")

            if hit_exit:
                final_exit_price = price_now
                pnl_full = final_exit_price - entry_price if entry_sig == 1 else entry_price - final_exit_price
                total_pnl = pnl_full

                fees = calculate_intraday_fees(entry_price, final_exit_price)
                print(f"Entry: {entry_price}, Exit: {final_exit_price}, Fees: {fees}, Net PnL: {net_pnl}")
                net_pnl = total_pnl - fees

                trades.append({
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'final_exit_price': final_exit_price,
                    'pnl_final': pnl_full,
                    'fees': fees,
                    'net_pnl': net_pnl,
                    'pnl': total_pnl,
                    'trade_type': 'Buy' if entry_sig == 1 else 'Short Sell',
                    'duration_min': duration
                })

                in_trade = False
                cooldown = COOLDOWN_BARS
                last_trade_exit = df.index[i]

    return trades
