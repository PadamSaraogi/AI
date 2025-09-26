# tickbus.py
import queue, itertools

# A single, process-wide queue and sequence that survive Streamlit reruns.
TICK_QUEUE = queue.Queue()
SEQ = itertools.count(1)

def put(tick):
    """Non-blocking put used by the websocket callback."""
    try:
        TICK_QUEUE.put(tick, block=False)
    except Exception:
        pass

def drain(max_items=10000):
    """Drain up to max_items items from the queue; returns a Python list."""
    out = []
    for _ in range(max_items):
        if TICK_QUEUE.empty():
            break
        out.append(TICK_QUEUE.get())
    return out

def next_seq():
    """Return the next sequence number (monotonic)."""
    return next(SEQ)
