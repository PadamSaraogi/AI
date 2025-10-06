"""
tickbus.py — robust event bus + bar aggregator (1s by default)

Features
- Thread-safe raw tick intake at any rate
- Deterministic OHLC/VWAP/Volume bars at fixed cadence (default 1s)
- Timestamp clamp to avoid future/past buckets stalling emission
- Safe emission (never crashes if a flush happens before first tick)
- Manual flush + lightweight debug logging
- Back-compat shims: put(), drain(), heartbeat(), TICK_QUEUE, next_seq()

Usage
------
import tickbus
tickbus.start_bar_aggregator(cadence_sec=1)
tickbus.put_raw_tick({"ts": time.time(), "price": 123.45, "size": 1.0})
bars = tickbus.drain_bars()
"""

from __future__ import annotations
import os
import time
import uuid
import queue
import threading
from typing import Dict, Any, List, Optional

# --------------------------- Diagnostics ---------------------------
BUS_ID = os.environ.get("TICKBUS_ID", str(uuid.uuid4())[:8])

_HEARTBEAT = 0
_HB_LOCK = threading.Lock()
_DEBUG = bool(int(os.environ.get("TICKBUS_DEBUG", "0")))

def enable_debug(on: bool = True):
    global _DEBUG
    _DEBUG = bool(on)

def heartbeat_inc(n: int = 1) -> None:
    global _HEARTBEAT
    with _HB_LOCK:
        _HEARTBEAT += n

def heartbeat_value() -> int:
    with _HB_LOCK:
        return _HEARTBEAT

def _log(msg: str):
    if _DEBUG:
        print(f"[tickbus {BUS_ID}] {msg}", flush=True)

# --------------------------- Queues ---------------------------
_RAW_TICKS: "queue.Queue[Dict[str, Any]]" = queue.Queue()
_BAR_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# --------------------------- Aggregator state ---------------------------
_BAR_LOCK = threading.Lock()
_RUNNING = False
_CADENCE = 1  # seconds

# Current bar accumulators
_curr_bucket: Optional[int] = None  # epoch seconds bucket start
_o = _h = _l = _c = None
_v_sum = 0.0
_pv_sum = 0.0

# Timestamp skew clamp (avoid stalling on future/past timestamps)
_TS_SKEW_SEC = 10

def _bucket_of(ts: float, cadence: int) -> int:
    sec = int(ts)
    return (sec // cadence) * cadence

def put_raw_tick(tick: Dict[str, Any]) -> None:
    """
    Enqueue a raw tick and update the current bar.

    tick:
      ts:    float|int epoch seconds (optional; uses time.time() if missing)
      price: float (required)
      size:  float (optional; default 1.0)
    """
    global _curr_bucket, _o, _h, _l, _c, _v_sum, _pv_sum

    if not isinstance(tick, dict) or "price" not in tick:
        return

    # normalize ts
    try:
        ts = float(tick.get("ts", time.time()))
    except Exception:
        ts = time.time()

    now = time.time()
    if abs(ts - now) > _TS_SKEW_SEC:
        ts = now  # clamp wild timestamps

    # normalize price/size
    try:
        px = float(str(tick["price"]).replace(",", ""))
    except Exception:
        return
    try:
        sz = float(str(tick.get("size", 1.0)).replace(",", ""))
    except Exception:
        sz = 1.0

    # store raw tick (optional consumer)
    _RAW_TICKS.put({"ts": ts, "price": px, "size": sz})

    bucket = _bucket_of(ts, _CADENCE)

    with _BAR_LOCK:
        # recover if current bucket somehow drifted far into the future
        if _curr_bucket is not None and _curr_bucket - now > 5 * _CADENCE:
            _log(f"recover: future bucket {_curr_bucket} >> now {now:.3f}; snap to now")
            _curr_bucket = _bucket_of(now, _CADENCE)
            _o = _h = _l = _c = None
            _v_sum = 0.0
            _pv_sum = 0.0

        if _curr_bucket is None:
            _curr_bucket = bucket
            _o = _h = _l = _c = px
            _v_sum = sz
            _pv_sum = px * sz
        elif bucket == _curr_bucket:
            _c = px
            _h = px if (_h is None or px > _h) else _h
            _l = px if (_l is None or px < _l) else _l
            _v_sum += sz
            _pv_sum += px * sz
        else:
            _emit_current_bar_locked()
            _curr_bucket = bucket
            _o = _h = _l = _c = px
            _v_sum = sz
            _pv_sum = px * sz

    heartbeat_inc()

def _emit_current_bar_locked() -> None:
    """Emit the current bar if we have prices; tolerate partial state. Must hold _BAR_LOCK."""
    global _curr_bucket, _o, _h, _l, _c, _v_sum, _pv_sum
    if _curr_bucket is None:
        return

    # If nothing priced this bucket yet, skip
    if _o is None and _h is None and _l is None and _c is None:
        return

    # Synthesize missing fields from the best available ref
    ref = _c if _c is not None else (_o if _o is not None else (_h if _h is not None else _l))
    o = _o if _o is not None else ref
    h = _h if _h is not None else ref
    l = _l if _l is not None else ref
    c = _c if _c is not None else ref

    vwap = (_pv_sum / _v_sum) if (_v_sum and _v_sum > 0) else c
    bar = {
        "ts": float(_curr_bucket),
        "end_ts": float(_curr_bucket + _CADENCE),
        "cadence": _CADENCE,
        "open": float(o),
        "high": float(h),
        "low":  float(l),
        "close": float(c),
        "vwap": float(vwap),
        "volume": float(_v_sum or 0.0),
    }
    _BAR_QUEUE.put(bar)
    _log(f"emit bar: [{int(bar['ts'])}->{int(bar['end_ts'])}] O={bar['open']:.4f} C={bar['close']:.4f} V={bar['volume']:.2f}")

def _bar_flusher_loop() -> None:
    """Flush bar on cadence boundary even without new ticks. Also recover future buckets."""
    global _curr_bucket, _o, _h, _l, _c, _v_sum, _pv_sum
    while True:
        time.sleep(0.05)
        now = time.time()
        with _BAR_LOCK:
            if _curr_bucket is None:
                continue

            # Recover from far-future bucket
            if _curr_bucket - now > 5 * _CADENCE:
                _log(f"flusher recover: future bucket {_curr_bucket} >> now {now:.3f}")
                _curr_bucket = _bucket_of(now, _CADENCE)
                _o = _h = _l = _c = None
                _v_sum = 0.0
                _pv_sum = 0.0
                continue

            if now >= (_curr_bucket + _CADENCE):
                # emit only if priced; else silently advance
                if not (_o is None and _h is None and _l is None and _c is None):
                    _emit_current_bar_locked()
                _curr_bucket = _bucket_of(now, _CADENCE)
                _o = _h = _l = _c = None
                _v_sum = 0.0
                _pv_sum = 0.0

def start_bar_aggregator(cadence_sec: int = 1) -> None:
    """Start the aggregator once (idempotent)."""
    global _RUNNING, _CADENCE
    if _RUNNING:
        return
    if cadence_sec <= 0:
        cadence_sec = 1
    _CADENCE = int(cadence_sec)
    t = threading.Thread(target=_bar_flusher_loop, daemon=True, name="tickbus_flusher")
    t.start()
    _RUNNING = True
    _log(f"aggregator started, cadence={_CADENCE}s")

def flush_now() -> bool:
    """Manually flush the current bar now. Returns True if a bar was emitted."""
    with _BAR_LOCK:
        before = _BAR_QUEUE.qsize()
        _emit_current_bar_locked()
        after = _BAR_QUEUE.qsize()
    return after > before

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

# --------------------------- Backward-compat shims ---------------------------
TICK_QUEUE = _RAW_TICKS  # alias for UIs

def put(x):
    """Legacy: accept tick(s) with varying shapes and push into aggregator."""
    batch = x if isinstance(x, list) else [x]
    for item in batch:
        if item is None:
            continue
        if isinstance(item, dict):
            # ts
            ts = item.get("ts")
            if ts is None:
                for tk in ("ltt","last_trade_time","exchange_time","trade_time","time","timestamp","datetime","created_at"):
                    v = item.get(tk)
                    if v:
                        try:
                            import pandas as _pd
                            tsv = _pd.to_datetime(v, utc=True, errors="coerce")
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
                    v = item.get(k)
                    if v not in (None, ""):
                        px = v; break
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
            try:
                put_raw_tick({"ts": time.time(), "price": float(item), "size": 1.0})
            except Exception:
                pass

def drain(max_items: int = 1000):
    """Legacy: return bars (new pipeline)."""
    return drain_bars(max_items)

def heartbeat():
    """Legacy alias."""
    return heartbeat_value()

_seq = 0
_seq_lock = threading.Lock()
def next_seq():
    global _seq
    with _seq_lock:
        _seq += 1
        return _seq

# --------------------------- Debug main ---------------------------
if __name__ == "__main__":
    enable_debug(True)
    print(f"[tickbus {BUS_ID}] demo")
    start_bar_aggregator(cadence_sec=1)
    import math
    base = 100.0
    t0 = time.time()
    # inject a few odd timestamps to prove clamping
    put_raw_tick({"ts": t0 + 3600, "price": base, "size": 1})
    for i in range(150):
        ts = time.time()
        px = base + 0.8 * math.sin(i / 8.0)
        put_raw_tick({"ts": ts, "price": px, "size": 1})
        time.sleep(0.02)
    flush_now()
    for b in drain_bars()[-10:]:
        print(bar_to_str(b))
    print("heartbeat:", heartbeat_value())
