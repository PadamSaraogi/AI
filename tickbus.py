# --- Live Trading Tab ---------------------------------------------------------
import time
import math
import numpy as np
import pandas as pd
import streamlit as st

import tickbus  # our bus + aggregator

# --- tiny indicator helpers ---------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def crossover(series_fast: pd.Series, series_slow: pd.Series) -> pd.Series:
    # +1 when fast crosses above slow, -1 when crosses below, 0 otherwise
    prev_fast = series_fast.shift(1)
    prev_slow = series_slow.shift(1)
    up = (prev_fast <= prev_slow) & (series_fast > series_slow)
    down = (prev_fast >= prev_slow) & (series_fast < series_slow)
    sig = pd.Series(0, index=series_fast.index, dtype=int)
    sig[up] = 1
    sig[down] = -1
    return sig

# --- trade engine (toy example: EMA crossover) -------------------------------
def maybe_trade_on_bar(bar: dict, state: dict):
    """
    Called once per emitted bar (e.g., per second).
    This is a toy strategy: EMA(5) vs EMA(20) on close.
    """
    price = bar["close"]
    ts = bar["end_ts"]  # bar end time

    # Append to history
    row = {
        "ts": pd.to_datetime(ts, unit="s"),
        "close": price,
        "vwap": bar["vwap"],
        "volume": bar["volume"],
    }
    state["df"] = pd.concat([state["df"], pd.DataFrame([row])], ignore_index=True)

    df = state["df"]
    # Indicators
    df["ema_fast"] = ema(df["close"], span=state["ema_fast_span"])
    df["ema_slow"] = ema(df["close"], span=state["ema_slow_span"])
    df["x"] = crossover(df["ema_fast"], df["ema_slow"])

    # Skip until we have both EMAs
    if len(df) < max(state["ema_fast_span"], state["ema_slow_span"]) + 2:
        return

    signal = int(df.iloc[-1]["x"])  # +1/-1/0 on this bar
    pos = state["position"]  # -1, 0, +1
    qty = state["qty_per_trade"]

    # simple execution simulator; replace with your broker call
    def fill(order_side: str, qty: float, px: float):
        # naive FIFO PnL for single-position logic
        if order_side == "BUY":
            state["cash"] -= px * qty
            state["position"] += qty
        else:
            state["cash"] += px * qty
            state["position"] -= qty
        state["trades"].append({"ts": row["ts"], "side": order_side, "qty": qty, "px": px})

    # entry/exit logic (flip to signal direction; flat when 0)
    if state["auto_trade"]:
        if signal > 0 and pos <= 0:
            # close short if any, then go long
            if pos < 0:
                fill("BUY", qty=abs(pos), px=price)
            fill("BUY", qty=qty, px=price)
        elif signal < 0 and pos >= 0:
            # close long if any, then go short
            if pos > 0:
                fill("SELL", qty=pos, px=price)
            fill("SELL", qty=qty, px=price)
        elif signal == 0:
            # optional: flatten on 0 signal (comment out if you prefer to keep position)
            if pos > 0:
                fill("SELL", qty=pos, px=price)
            elif pos < 0:
                fill("BUY", qty=abs(pos), px=price)

    # mark-to-market
    pos_val = state["position"] * price
    state["equity"] = state["cash"] + pos_val
    state["last_price"] = price

def render_live_trading_tab():
    st.header("📡 Live Trading")

    # --- one-time init --------------------------------------------------------
    if "lt_init" not in st.session_state:
        st.session_state.lt_init = True
        st.session_state.lt_state = {
            "df": pd.DataFrame(columns=["ts", "close", "vwap", "volume"]).astype(
                {"ts": "datetime64[ns]", "close": "float64", "vwap": "float64", "volume": "float64"}
            ),
            "position": 0.0,
            "cash": 1_000_000.0,  # starting cash for PnL calc
            "equity": 1_000_000.0,
            "trades": [],
            "last_price": None,
            "auto_trade": False,
            "qty_per_trade": 1.0,
            "ema_fast_span": 5,
            "ema_slow_span": 20,
        }
        # Start the bar aggregator at 1-second cadence
        tickbus.start_bar_aggregator(cadence_sec=1)

    state = st.session_state.lt_state

    # --- controls -------------------------------------------------------------
    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        state["auto_trade"] = st.toggle("Auto Trade", value=state["auto_trade"])
    with colB:
        state["qty_per_trade"] = st.number_input("Qty per trade", value=float(state["qty_per_trade"]), min_value=0.0, step=1.0)
    with colC:
        state["ema_fast_span"] = st.number_input("EMA Fast", value=int(state["ema_fast_span"]), min_value=2, step=1)
    with colD:
        state["ema_slow_span"] = st.number_input("EMA Slow", value=int(state["ema_slow_span"]), min_value=3, step=1)

    st.caption(f"tickbus id: **{tickbus.BUS_ID}** | cadence: **1s** | heartbeat: **{tickbus.heartbeat_value()}**")

    # --- Simulation mode (optional) -------------------------------------------
    with st.expander("Simulation Mode (no broker)", expanded=False):
        sim_on = st.checkbox("Run price simulator (sine + noise)", value=False)
        sim_speed = st.slider("Simulator ticks per second", 1.0, 50.0, 15.0, 0.5)

        if sim_on:
            # emit a short burst each render; Streamlit reruns frequently
            t0 = time.time()
            for i in range(8):
                phase = (t0 + i / sim_speed) * 0.7
                price = 100.0 + 0.6 * math.sin(phase) + np.random.normal(0, 0.03)
                tickbus.put_raw_tick({"ts": time.time(), "price": price, "size": 1})

    # --- processing loop: drain bars once per render --------------------------
    bars = tickbus.drain_bars()
    if bars:
        for b in bars:
            maybe_trade_on_bar(b, state)

    # --- charts ---------------------------------------------------------------
    df = state["df"].copy()
    if not df.empty:
        df = df.set_index("ts")
        df["ema_fast"] = df["close"].ewm(span=state["ema_fast_span"], adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=state["ema_slow_span"], adjust=False).mean()

        st.line_chart(df[["close", "ema_fast", "ema_slow"]])

    # --- positions & PnL ------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Last Price", f"{state['last_price']:.4f}" if state["last_price"] else "—")
    with c2:
        st.metric("Position (qty)", f"{state['position']:.2f}")
    with c3:
        st.metric("Equity (sim)", f"{state['equity']:.2f}")

    # --- trades table ---------------------------------------------------------
    if state["trades"]:
        tdf = pd.DataFrame(state["trades"])
        tdf = tdf.sort_values("ts", ascending=False).reset_index(drop=True)
        st.dataframe(tdf.head(50), use_container_width=True)

    # --- gentle auto-refresh to keep UI updated -------------------------------
    # NOTE: trading decisions are NOT tied to this timer; they run per emitted bar.
    st.toast("Live loop tick", icon="⏱️")
    st.experimental_rerun()
# -----------------------------------------------------------------------------


# If you use tabs elsewhere:
# tab1, tab2, tab3 = st.tabs(["Live Trading", "Backtest", "Settings"])
# with tab1:
#     render_live_trading_tab()
