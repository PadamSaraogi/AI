from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import json
import io
import os
import sys

# Add root directory to path to import backtest and tickbus
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest
import tickbus

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/backtest")
async def run_backtest(
    signal_files: List[UploadFile] = File(...),
    grid_files: List[UploadFile] = File(...),
    total_capital: float = Form(...),
    risk_per_trade: float = Form(...)
):
    # Dictionary to store stock data
    stock_data = {}
    
    # Helper to extract symbol from filename
    def extract_symbol(fname):
        return fname.split('_')[-1].split('.')[0].lower()

    # Process signal files
    signal_dict = {extract_symbol(f.filename): f for f in signal_files}
    grid_dict = {extract_symbol(f.filename): f for f in grid_files}
    
    common_symbols = set(signal_dict.keys()) & set(grid_dict.keys())
    
    results = {}
    capital_per_stock = total_capital // len(common_symbols) if common_symbols else 0
    
    for symbol in common_symbols:
        sig_content = await signal_dict[symbol].read()
        grid_content = await grid_dict[symbol].read()
        
        df_signals = pd.read_csv(io.BytesIO(sig_content), parse_dates=['datetime'])
        df_signals.set_index('datetime', inplace=True)
        # Assuming grid search results are not needed for the core simulation but can be returned
        
        trades_df, equity_curve = backtest.run_backtest_simulation(
            df_signals,
            starting_capital=capital_per_stock,
            risk_per_trade=risk_per_trade / 100
        )
        
        # Convert dataframes to JSON serializable format
        results[symbol] = {
            "trades": trades_df.to_dict(orient="records"),
            "equity_curve": equity_curve.to_dict(),
            "metrics": {
                "total_trades": len(trades_df),
                "net_pnl": float(trades_df['net_pnl'].sum()) if not trades_df.empty else 0.0,
                "win_rate": float((trades_df['net_pnl'] > 0).mean()) if not trades_df.empty else 0.0,
                "final_capital": float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else capital_per_stock
            }
        }

    return results

@app.get("/api/live/bars")
def get_live_bars():
    # This would normally pull from a stateful aggregator.
    # On Vercel, we might need an external store if we want persistent data.
    # For now, we'll return what's in the current process memory.
    bars = tickbus.drain_bars()
    return {"bars": bars}

@app.post("/api/live/tick")
def post_tick(tick: dict):
    tickbus.put_raw_tick(tick)
    return {"status": "success"}
