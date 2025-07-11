import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from backtest import run_backtest_simulation

st.set_page_config(layout="wide")
st.title("📈 Personal Trading Dashboard")

# Upload inputs
csv_file = st.file_uploader("📂 Upload your indicator CSV file", type="csv")
model_file = st.file_uploader("🧠 Upload your trained ML model (.pkl)", type="pkl")

if csv_file and model_file:
    # Load data
    df = pd.read_csv(csv_file, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    model = joblib.load(model_file)

    st.success("✅ Data and model loaded successfully.")

    # Prepare features
    features = ['ema_20', 'ema_50', 'ATR', 'ADX14', 'RSI', 'bb_width',
                'volume_spike_ratio', 'return_1h', 'hour_of_day']
    X = df[features]
    proba = model.predict_proba(X)
    df['predicted_label'] = model.predict(X)
    df['confidence'] = np.max(proba, axis=1)

    # Generate signals
    threshold = 0.6
    signals = []
    for i in range(len(df)):
        sig = 0
        label = df['predicted_label'].iat[i]
        conf = df['confidence'].iat[i]
        rsi = df['RSI'].iat[i]
        ema = df['ema_50'].iat[i]
        price = df['close'].iat[i]

        if label == 1 and conf >= threshold and price > ema * 0.99 and rsi > 30:
            sig = 1
        elif label == -1 and conf >= threshold and price < ema * 1.01 and rsi < 70:
            sig = -1
        signals.append(sig)

    df['signal'] = signals
    df['position'] = df['signal'].replace(0, np.nan).ffill()

    st.subheader("📋 Signal Preview")
    st.dataframe(df[['close', 'signal', 'confidence']].tail(20))

    st.subheader("📊 Signal Chart")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['close'], label='Close Price', color='gray')
    ax.plot(df[df['signal'] == 1].index, df['close'][df['signal'] == 1], '^', color='green', label='Buy')
    ax.plot(df[df['signal'] == -1].index, df['close'][df['signal'] == -1], 'v', color='red', label='Sell')
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔙 Backtest Results")
    trades = run_backtest_simulation(df)
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['cumulative_pnl'].cummax()

        st.subheader("📈 Equity Curve")
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.plot(trades_df['exit_time'], trades_df['cumulative_pnl'], color='blue')
        ax2.set_ylabel("Cumulative PnL")
        st.pyplot(fig2)

        st.subheader("📊 Backtest Summary")
        summary = {
            'Total Legs': len(trades_df),
            'Win Rate (%)': 100 * (trades_df['pnl'] > 0).mean(),
            'Total PnL': trades_df['pnl'].sum(),
            'Avg PnL per Leg': trades_df['pnl'].mean(),
            'Max Drawdown': trades_df['drawdown'].min()
        }
        st.dataframe(pd.DataFrame([summary]))
    else:
        st.warning("No trades generated.")
else:
    st.info("Please upload both a CSV and PKL file to begin.")
