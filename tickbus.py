# tickbus.py
import queue, itertools, uuid, threading

# A single, process-wide queue and sequence that survive Streamlit reruns.
TICK_QUEUE = queue.Queue()
SEQ = itertools.count(1)

# Helpful to confirm both threads use the same bus
BUS_ID = str(uuid.uuid4())[:8]

# Heartbeat counter (increments whenever a tick is enqueued)
_HB_LOCK = threading.Lock()
_HEARTBEAT = 0

def put(tick) -> None:
    """Non-blocking put used by the websocket callback."""
    global _HEARTBEAT
    try:
        TICK_QUEUE.put(tick, block=False)
        with _HB_LOCK:
            _HEARTBEAT = (_HEARTBEAT + 1) % 1_000_000_000
    except Exception:
        pass

def drain(max_items: int = 10000):
    """Drain up to max_items items from the queue; returns a Python list."""
    out = []
    for _ in range(max_items):
        if TICK_QUEUE.empty():
            break
        out.append(TICK_QUEUE.get())
    return out

def next_seq() -> int:
    """Return the next sequence number (monotonic)."""
    return next(SEQ)

def heartbeat() -> int:
    """A monotonically increasing counter of enqueued ticks."""
    with _HB_LOCK:
        return _HEARTBEAT
