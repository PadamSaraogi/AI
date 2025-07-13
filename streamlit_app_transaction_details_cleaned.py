import pandas as pd
import numpy as np

def run_backtest_simulation(df, trail_mult=2.0, time_limit=16, adx_target_mult=2.5):
    trades = []
    in_trade = False
    cooldown = 0
    COOLDOWN_BARS = 2
    STOP_MULT = 1.0

    for i in range(1, len(df)):
        if cooldown > 0:
            cooldown -= 1
            continue

        sig = df['signal'].iat[i]
        price = df['close'].iat[i]
        atr = df['ATR'].iat[i]
        adx = df['ADX14'].iat[i]

        if not in_trade and sig != 0:
            entry_price = price
            entry_sig = sig  # +1 for long, -1 for short
            stop_price = entry_price - STOP_MULT * atr * entry_sig
            tp_half = entry_price + 1.0 * atr * entry_sig

            if adx < 20:
                target_mult = 2.0
            elif adx < 30:
                target_mult = 2.5
            else:
                target_mult = adx_target_mult

            tp_full = entry_price + target_mult * atr * entry_sig
            trail_price = entry_price
            in_trade = True
            entry_idx = i
            half_closed = False
            pnl_half = 0.0  # initialize

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

            # First Half Exit (TP1)
            if not half_closed:
                hit_tp1 = (entry_sig > 0 and price_now >= tp_half) or (entry_sig < 0 and price_now <= tp_half)
                if hit_tp1:
                    if entry_sig == 1:
                        pnl_half = 0.5 * (price_now - entry_price)
                    else:
                        pnl_half = 0.5 * (entry_price - price_now)
                    half_closed = True
                    continue

            # Final Exit Conditions
            hit_exit = (
                (entry_sig > 0 and (price_now <= stop_price or price_now >= tp_full or price_now <= trailing_stop)) or
                (entry_sig < 0 and (price_now >= stop_price or price_now <= tp_full or price_now >= trailing_stop)) or
                duration >= time_limit
            )

            if hit_exit:
                if entry_sig == 1:
                    pnl_full = 0.5 * (price_now - entry_price)
                else:
                    pnl_full = 0.5 * (entry_price - price_now)

                total_pnl = pnl_half + pnl_full if half_closed else pnl_full

                trades.append({
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': price_now,
                    'pnl': total_pnl
                })

                in_trade = False
                cooldown = COOLDOWN_BARS

    return trades
