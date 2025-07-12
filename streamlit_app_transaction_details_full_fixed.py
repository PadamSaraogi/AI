
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from backtest import run_backtest_simulation

st.set_page_config(layout="wide")
st.title("📈 Enhanced Trading Dashboard")

csv_file = st.file_uploader("📂 Upload your indicator CSV file", type="csv")
model_file = st.file_uploader("🧠 Upload your trained ML model (.pkl)", type="pkl")

if csv_file and model_file:
    df = pd.read_csv(csv_file, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    model = joblib.load(model_file)

    st.success("✅ Model and data loaded.")
    st.write(f"CSV: `{csv_file.name}` | Model: `{model_file.name}`")

    features = ['ema_20', 'ema_50', 'ATR', 'ADX14', 'RSI', 'bb_width',
                'volume_spike_ratio', 'return_1h', 'hour_of_day']
    X = df[features]
    proba = model.predict_proba(X)
    df['predicted_label'] = model.predict(X)
    df['confidence'] = np.max(proba, axis=1)

    if hasattr(model, 'feature_importances_'):
        st.subheader("🧠 Feature Importance")
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        st.bar_chart(feat_imp)

    threshold = st.slider("🎚 Confidence Threshold", 0.0, 1.0, 0.6, 0.01)
    signals = []
    transaction_details = []

    for i in range(len(df)):
        sig = 0
        label = df['predicted_label'].iat[i]
        conf = df['confidence'].iat[i]
        rsi = df['RSI'].iat[i]
        ema = df['ema_50'].iat[i]
        price = df['close'].iat[i]

        # Logic for Buy, Sell, Short Signal
        if label == 1 and conf >= threshold and price > ema * 0.99 and rsi > 30:
            sig = 1
            transaction_details.append(f"Buy at {price} on {df.index[i]}")
        elif label == -1 and conf >= threshold and price < ema * 1.01 and rsi < 70:
            sig = -1
            transaction_details.append(f"Sell at {price} on {df.index[i]}")
        elif label == 0 and conf >= threshold:
            sig = 0
            transaction_details.append(f"Short sell at {price} on {df.index[i]}")
        signals.append(sig)

    df['signal'] = signals
    df['position'] = df['signal'].replace(0, np.nan).ffill()

    # Calculate the duration of each trade in minutes
    df['duration'] = (df.index - df.index.shift(1)).astype('timedelta64[m]')

    tab1, tab2, tab3, tab4 = st.tabs(["🧾 Signals", "📊 Charts", "📈 Backtest", "📋 Stats"])

    with tab1:
        st.subheader("📋 Recent Signals with Transaction Info")
        st.dataframe(df[['close', 'signal', 'confidence', 'duration']].tail(20))
        for transaction in transaction_details:
            st.write(transaction)

    with tab2:
        st.subheader("📈 Signal Chart with Transactions")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df['close'], label='Close Price', color='gray')
        ax.plot(df[df['signal'] == 1].index, df['close'][df['signal'] == 1], '^', color='green', label='Buy')
        ax.plot(df[df['signal'] == -1].index, df['close'][df['signal'] == -1], 'v', color='red', label='Sell')
        ax.plot(df[df['signal'] == 0].index, df['close'][df['signal'] == 0], 'o', color='orange', label='Short')
        ax.legend()
        st.pyplot(fig)

    with tab3:
        trades = run_backtest_simulation(df)
        trades_df = pd.DataFrame(trades)

        if not trades_df.empty:
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['cumulative_pnl'].cummax()

            st.subheader("📈 Equity Curve")
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            ax2.plot(trades_df['exit_time'], trades_df['cumulative_pnl'], color='blue')
            st.pyplot(fig2)

            st.download_button("📥 Download Trades", trades_df.to_csv().encode(), "trades.csv", "text/csv")

    with tab4:
        if not trades_df.empty:
            sharpe = trades_df['pnl'].mean() / trades_df['pnl'].std() * np.sqrt(252) if trades_df['pnl'].std() > 0 else 0
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total PnL", f"{trades_df['pnl'].sum():.2f}")
            col2.metric("Win Rate", f"{(trades_df['pnl'] > 0).mean() * 100:.1f}%")
            col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col4.metric("Trades", f"{len(trades_df)}")
            col5.metric("Avg Hold (min)", f"{trades_df['duration'].mean():.1f}")

else:
    st.info("📁 Please upload both a CSV and PKL file to begin.")
