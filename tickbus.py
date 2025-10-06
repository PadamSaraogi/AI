"""
tickbus.py
Event bus + bar aggregator for live ticks.

- Thread-safe raw tick ingestion at any rate.
- Deterministic bar aggregation at fixed cadence (default 1 second).
- Emits OHLC, VWAP, Volume bars you can drain in your trading loop.
- Diagnostics: BUS_ID + heartbeat counter.
- Backward-compat shims for legacy calls: put(), drain(), heartbeat(), TICK_QUEUE, next_seq().

Usage
------
import tickbus

tickbus.start_bar_aggregator(cadence_sec=1)  # start once (1-second bars)

# push raw ticks from your broker callback:
tickbus.put_raw_tick({"ts": time.time(), "price": 123.45, "size": 1.0})

# in your processing loop (UI or worker):
bars = tickbus.drain_bars()
for bar in bars:
    # bar: {ts, end_ts, cadence, open, high, low, close, vwap, volume}
    ...
"""

from __future__ import annotations
import os
import time
import json
import uuid
import queue
import threading
from typing import Dict, Any, List, Optional

# ---------------------------
# Diagnostics
# ---------------------------
BUS_ID = os.environ.get("TICKBUS_ID", str(uuid.uuid4())[:8])

_HEARTBEAT = 0
_HB_LOCK = threading.Lock()

def heartbeat_inc(n: int = 1) -> None:
    global _HEARTBEAT
    with _HB_LOCK:
        _HEARTBEAT += n

def heartbeat_value() -> int:
    with _HB_LOCK:
        return _HEARTBEAT

# ---------------------------
# Queues
# ---------------------------
_RAW_TICKS: "queue.Queue[Dict[str, Any]]" = queue.Queue()
_BAR_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# ---------------------------
# Aggregator state
# ---------------------------
_BAR_LOCK = threading.Lock()
_RUNNING = False
_CADENCE = 1  # seconds

# Current bar accumulators
_curr_bucket: Optional[int] = None  # epoch seconds bucket start
_o = _h = _l = _c = None
_v_sum = 0.0
_pv_sum = 0.0

def _bucket_of(ts: float, cadence: int) -> int:
    sec = int(ts)
    return (sec // cadence) * cadence

def put_raw_tick(tick: Dict[str, Any]) -> None:
    """
    Public: enqueue a raw tick and update aggregator.

    tick:
      ts:   float|int epoch seconds (optional; uses time.time() if missing)
      price: float (required)
      size:  float (optional; default 1.0)
    """
    global _curr_bucket, _o, _h, _l, _c, _v_sum, _pv_sum

    if not isinstance(tick, dict) or "price" not in tick:
        return

    try:
        ts = float(tick.get("ts", time.time()))
    except Exception:
        ts = time.time()

    # normalize price/size
    px = tick["price"]
    try:
        px = float(str(px).replace(",", ""))
    except Exception:
        return

    sz = tick.get("size", 1.0)
    try:
        sz = float(str(sz).replace(",", ""))
    except Exception:
        sz = 1.0

    # Store raw tick (optional consumer: charts / debug)
    _RAW_TICKS.put({"ts": ts, "price": px, "size": sz})

    # Aggregate into cadence bucket
    bucket = _bucket_of(ts, _CADENCE)

    with _BAR_LOCK:
        if _curr_bucket is None:
            # initialize bar
            _curr_bucket = bucket
            _o = _h = _l = _c = px
            _v_sum = sz
            _pv_sum = px * sz
        elif bucket == _curr_bucket:
            # update current bar
            _c = px
            _h = px if (_h is None or px > _h) else _h
            _l = px if (_l is None or px < _l) else _l
            _v_sum += sz
            _pv_sum += px * sz
        else:
            # boundary crossed: emit previous, start new
            _emit_current_bar_locked()
            _curr_bucket = bucket
            _o = _h = _l = _c = px
            _v_sum = sz
            _pv_sum = px * sz

    heartbeat_inc()

def _emit_current_bar_locked() -> None:
    """Emit the current bar. Must be called under _BAR_LOCK."""
    global _curr_bucket, _o, _h, _l, _c, _v_sum, _pv_sum
    if _curr_bucket is None or _c is None:
        return
    vwap = (_pv_sum / _v_sum) if _v_sum > 0 else _c
    bar = {
        "ts": float(_curr_bucket),   # start time of the bar
        "end_ts": float(_curr_bucket + _CADENCE),
        "cadence": _CADENCE,
        "open": float(_o),
        "high": float(_h),
        "low": float(_l),
        "close": float(_c),
        "vwap": float(vwap),
        "volume": float(_v_sum),
    }
    _BAR_QUEUE.put(bar)

def _bar_flusher_loop() -> None:
    """
    Background thread: on cadence boundary, flush bar even if no new ticks.
    Emits a 'carry' bar for quiet intervals (volume may be zero).
    """
    global _curr_bucket, _o, _h, _l, _c, _v_sum, _pv_sum
    while True:
        time.sleep(0.05)  # fine-grained boundary detection
        now = time.time()
        with _BAR_LOCK:
            if _curr_bucket is None:
                continue
            if now >= (_curr_bucket + _CADENCE):
                _emit_current_bar_locked()
                # advance to next bucket; leave accumulators empty until next tick
                next_bucket = _bucket_of(now, _CADENCE)
                _curr_bucket = next_bucket
                _o = _h = _l = _c = None
                _v_sum = 0.0
                _pv_sum = 0.0

def start_bar_aggregator(cadence_sec: int = 1) -> None:
    """
    Start the aggregator once (idempotent).
    cadence_sec: integer seconds per bar (e.g., 1, 5, 60)
    """
    global _RUNNING, _CADENCE
    if _RUNNING:
        return
    if cadence_sec <= 0:
        cadence_sec = 1
    _CADENCE = int(cadence_sec)
    t = threading.Thread(target=_bar_flusher_loop, daemon=True, name="tickbus_flusher")
    t.start()
    _RUNNING = True

def drain_bars(max_items: int = 1000) -> List[Dict[str, Any]]:
    """Drain emitted bars (OHLC/VWAP)."""
    out: List[Dict[str, Any]] = []
    for _ in range(max_items):
        try:
            out.append(_BAR_QUEUE.get_nowait())
        except queue.Empty:
            break
    return out

def drain_raw(max_items: int = 2000) -> List[Dict[str, Any]]:
    """Drain raw ticks (for charts/debug)."""
    out: List[Dict[str, Any]] = []
    for _ in range(max_items):
        try:
            out.append(_RAW_TICKS.get_nowait())
        except queue.Empty:
            break
    return out

def bar_to_str(bar: Dict[str, Any]) -> str:
    return (f"[{int(bar['ts'])}->{int(bar['end_ts'])}|{bar['cadence']}s] "
            f"O:{bar['open']:.2f} H:{bar['high']:.2f} L:{bar['low']:.2f} "
            f"C:{bar['close']:.2f} VWAP:{bar['vwap']:.2f} V:{bar['volume']:.2f}")

# ---------------------------
# Backward-compat shims
# ---------------------------
# Expose a queue named TICK_QUEUE (some UIs display it)
TICK_QUEUE = _RAW_TICKS  # alias

def put(x):
    """
    Legacy: accept a single tick or a list of ticks and push into the aggregator.
    Tries to infer price/time/size from common keys.
    """
    batch = x if isinstance(x, list) else [x]
    for item in batch:
        if item is None:
            continue
        if isinstance(item, dict):
            # ts
            ts = item.get("ts")
            if ts is None:
                # try common time keys
                for tk in ("ltt","last_trade_time","exchange_time","trade_time",
                           "time","timestamp","datetime","created_at"):
                    if tk in item and item[tk]:
                        try:
                            import pandas as _pd
                            tsv = _pd.to_datetime(item[tk], utc=True, errors="coerce")
                            ts = tsv.timestamp() if _pd.notna(tsv) else None
                        except Exception:
                            ts = None
                        break
            if ts is None:
                ts = time.time()

            # price
            px = item.get("price")
            if px is None:
                for k in ("last","Last","LAST","last_traded_price","LastTradedPrice","lastTradedPrice",
                          "ltp","LTP","lastPrice","LastPrice","close","Close","price","Price"):
                    if k in item and item[k] not in (None, ""):
                        px = item[k]; break
            if px is None:
                continue
            try:
                px = float(str(px).replace(",", ""))
            except Exception:
                continue

            # size
            sz = item.get("size") or item.get("volume") or item.get("qty") or 1.0
            try:
                sz = float(str(sz).replace(",", ""))
            except Exception:
                sz = 1.0

            put_raw_tick({"ts": ts, "price": px, "size": sz})

        else:
            # number/string → price; now time
            try:
                put_raw_tick({"ts": time.time(), "price": float(item), "size": 1.0})
            except Exception:
                pass

def drain(max_items: int = 1000):
    """Legacy: previously drained raw ticks; now return bars for compatibility with new pipeline."""
    return drain_bars(max_items)

def heartbeat():
    """Legacy alias."""
    return heartbeat_value()

# Simple global sequence for any code that calls tickbus.next_seq()
_seq = 0
_seq_lock = threading.Lock()
def next_seq():
    global _seq
    with _seq_lock:
        _seq += 1
        return _seq


# ---------------------------
# Optional: debug main
# ---------------------------
if __name__ == "__main__":
    print(f"[tickbus {BUS_ID}] demo")
    start_bar_aggregator(cadence_sec=1)
    import math
    base = 100.0
    t0 = time.time()
    for i in range(250):
        ts = t0 + i * 0.02
        px = base + 0.8 * math.sin(i / 8.0)
        put_raw_tick({"ts": ts, "price": px, "size": 1})
        time.sleep(0.01)
    for b in drain_bars():
        print(bar_to_str(b))
    print("heartbeat:", heartbeat_value())
