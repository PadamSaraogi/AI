import pandas as pd
import numpy as np
from datetime import timedelta

def run_backtest_simulation(df, trail_mult=2.0, time_limit=10, adx_target_mult=2.5):
    """
    Simulates trading based on signals and tracks performance.
    
    Parameters:
    - df: DataFrame containing the stock data and signals
    - trail_mult: Multiplier for trailing stop loss
    - time_limit: The maximum duration (in minutes) for each trade
    - adx_target_mult: Multiplier to define ADX target for filtering trades
    
    Returns:
    - List of executed trades with PnL and other details.
    """
    trades = []
    in_trade = False
    entry_price = 0
    entry_time = None
    stop_loss = 0
    take_profit = 0
    trade_type = None
    trade_entry_index = None
    
    for i in range(1, len(df)):
        signal = df['signal'].iat[i]
        price = df['close'].iat[i]
        current_time = df.index[i]

        # If we are in a trade, check for exit conditions
        if in_trade:
            time_elapsed = (current_time - entry_time).total_seconds() / 60  # Time in minutes
            
            # Check if we need to exit due to time limit
            if time_elapsed > time_limit:
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': price - entry_price,
                    'trade_type': trade_type,
                })
                in_trade = False
                continue  # Skip further checks, and start a new trade
            
            # Check if price hits the trailing stop or take profit
            if trade_type == 1:  # Buy trade
                stop_loss = entry_price - (trail_mult * (entry_price - price))
                take_profit = entry_price + (entry_price * 0.02)  # Target 2% profit for simplicity
                
                if price <= stop_loss or price >= take_profit:  # Exit condition
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': price - entry_price,
                        'trade_type': trade_type,
                    })
                    in_trade = False

            elif trade_type == -1:  # Sell trade
                stop_loss = entry_price + (trail_mult * (price - entry_price))
                take_profit = entry_price - (entry_price * 0.02)  # Target 2% profit for simplicity
                
                if price >= stop_loss or price <= take_profit:  # Exit condition
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': entry_price - price,
                        'trade_type': trade_type,
                    })
                    in_trade = False
        
        # If we are not in a trade, check for entry conditions
        if not in_trade:
            if signal == 1:  # Buy signal
                entry_price = price
                entry_time = current_time
                trade_type = 1
                in_trade = True  # Enter trade

            elif signal == -1:  # Sell signal
                entry_price = price
                entry_time = current_time
                trade_type = -1
                in_trade = True  # Enter trade

    # Return the list of trades made during the backtest
    return trades


def backtest_strategy(input_file, model, features, ml_threshold=0.5, trail_mult=2.0, time_limit=10, adx_target_mult=2.5):
    """
    Runs the backtest using the optimized parameters.
    
    Parameters:
    - input_file: Path to the CSV file with stock data
    - model: The trained ML model for predictions
    - features: The list of features to be used in the model
    - ml_threshold: Minimum confidence for a valid signal
    - trail_mult: Multiplier for trailing stop loss
    - time_limit: Maximum duration for trades in minutes
    - adx_target_mult: ADX multiplier for trade filtering
    
    Returns:
    - backtest_results: A DataFrame containing the trades made and their performance
    """
    # Load stock data
    df = pd.read_csv(input_file, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)  # Use datetime as the index

    # Make predictions using the model
    X = df[features]
    proba = model.predict_proba(X)
    df['predicted_label'] = model.predict(X)
    df['confidence'] = np.max(proba, axis=1)
    
    # Apply filtering to the signals based on ML confidence
    df['signal'] = np.where((df['confidence'] >= ml_threshold) & (df['predicted_label'] != 0), df['predicted_label'], 0)
    
    # Run backtest simulation
    trades = run_backtest_simulation(df, trail_mult=trail_mult, time_limit=time_limit, adx_target_mult=adx_target_mult)

    # If no trades were made, return an empty dataframe
    if not trades:
        print("No trades executed during the backtest.")
        return pd.DataFrame()

    # Convert the list of trades to a DataFrame for easier analysis
    trades_df = pd.DataFrame(trades)
    trades_df['pnl'] = trades_df['pnl'].astype(float)  # Ensure PnL is numeric
    
    # Calculate additional metrics like win rate and total profit
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    total_pnl = trades_df['pnl'].sum()
    max_drawdown = trades_df['pnl'].cumsum().min()
    
    print(f"Backtest Results: Win Rate = {win_rate:.2f}%, Total PnL = {total_pnl:.2f}, Max Drawdown = {max_drawdown:.2f}")
    
    return trades_df


# Example usage:
input_file = '5m_signals_enhanced_BEL.csv'  # Path to the CSV data file
model_file = 'trade_filter_model_BEL.pkl'  # Path to the trained model file
features = ['ema_20', 'ema_50', 'ATR', 'ADX14', 'RSI', 'bb_width', 'volume_spike_ratio', 'return_1h', 'hour_of_day']

# Load the trained model
import joblib
model = joblib.load(model_file)

# Run backtest
backtest_results = backtest_strategy(input_file, model, features)

# Display backtest results if available
if not backtest_results.empty:
    print(backtest_results.head())
