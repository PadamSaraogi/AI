import pandas as pd

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

def run_backtest_simulation(
    df,
    trail_mult=2.0, 
    time_limit=16, 
    adx_target_mult=2.5,
    starting_capital=100_000,  # New: user-controllable via Streamlit
    risk_per_trade=0.01        # New: user-controllable via Streamlit
):
    trades = []
    in_trade = False
    COOLDOWN_BARS = 0
    STOP_MULT = 1.0
    EXIT_TIME = pd.to_datetime("15:25:00").time()
    REENTER_TIME = pd.to_datetime("09:25:00").time()
    last_trade_exit = None
    entry_price = None
    entry_sig = None
    stop_price = None
    tp_full = None
    trail_price = None
    entry_idx = None
    position_size = None  # qty for the trade
    capital = starting_capital

    for i in range(1, len(df)):
        sig = df['signal'].iat[i]
        price = df['close'].iat[i]
        atr = df['ATR'].iat[i]
        adx = df['ADX14'].iat[i]
        trade_time = df.index[i].time()
        trade_date = df.index[i].date()

        # Prevent trades before 9:20am (add this line if needed)
        ALLOW_TRADING_AFTER = pd.to_datetime("09:20:00").time()
        if trade_time <= ALLOW_TRADING_AFTER:
            continue

        # Optional: Restrict new trades to after previous day's exit.
        if last_trade_exit and trade_time >= REENTER_TIME and trade_date > last_trade_exit.date():
            if not in_trade and sig != 0:
                entry_price = price
                entry_sig = sig
                stop_price = entry_price - STOP_MULT * atr * entry_sig
                tp_full = entry_price + adx_target_mult * atr * entry_sig
                trail_price = entry_price

                # ATR-based position sizing
                trade_risk = abs(entry_price - stop_price)
                risk_amount = capital * risk_per_trade
                qty = max(1, int(risk_amount // trade_risk)) if trade_risk > 0 else 1

                position_size = qty
                in_trade = True
                entry_idx = i
                continue

        if not in_trade and sig != 0:
            entry_price = price
            entry_sig = sig
            stop_price = entry_price - STOP_MULT * atr * entry_sig
            tp_full = entry_price + adx_target_mult * atr * entry_sig
            trail_price = entry_price

            # ATR-based position sizing
            trade_risk = abs(entry_price - stop_price)
            risk_amount = capital * risk_per_trade
            qty = max(1, int(risk_amount // trade_risk)) if trade_risk > 0 else 1

            position_size = qty
            in_trade = True
            entry_idx = i
            continue

        if in_trade:
            duration = i - entry_idx
            price_now = price
            atr_now = atr

            # Trailing stop management
            if entry_sig > 0:
                trail_price = max(trail_price, price_now)
                trailing_stop = trail_price - trail_mult * atr_now
            else:
                trail_price = min(trail_price, price_now)
                trailing_stop = trail_price + trail_mult * atr_now

            hit_exit = (
                (entry_sig > 0 and (
                    price_now <= stop_price or price_now >= tp_full or price_now <= trailing_stop
                )) or
                (entry_sig < 0 and (
                    price_now >= stop_price or price_now <= tp_full or price_now >= trailing_stop
                )) or
                duration >= time_limit
            )

            if trade_time >= EXIT_TIME:
                hit_exit = True

            if hit_exit:
                final_exit_price = price_now
                pnl_full = ((final_exit_price - entry_price) if entry_sig == 1 else (entry_price - final_exit_price)) * position_size

                fees = calculate_intraday_fees(entry_price, final_exit_price, position_size)
                net_pnl = pnl_full - fees

                capital += net_pnl

                trades.append({
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'final_exit_price': final_exit_price,
                    'pnl_final': pnl_full,
                    'fees': fees,
                    'net_pnl': net_pnl,
                    'pnl': pnl_full,
                    'trade_type': 'Buy' if entry_sig == 1 else 'Short Sell',
                    'duration_min': duration,
                    'position_size': position_size,
                    'capital_after_trade': capital
                })

                in_trade = False
                last_trade_exit = df.index[i]
                position_size = None  # reset position

    return trades

# Test block/example (optional):
if __name__ == "__main__":
    # Example usage:
    # df = pd.read_csv("5m_signals_enhanced_XYZ.csv", parse_dates=['datetime'])
    # df.set_index('datetime', inplace=True)
    # trades = run_backtest_simulation(df)
    pass
