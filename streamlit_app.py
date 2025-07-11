# Create an updated version of streamlit_app.py with all the requested enhancements
updated_streamlit_app_code = '''\
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from backtest import run_backtest_simulation

st.set_page_config(layout="wide")
st.title("📈 Upload-Based Trading Dashboard")

# File upload UI
csv_file = st.file_uploader("📂 Upload your indicator CSV file", type="csv")
model_file = st.file_uploader("🧠 Upload your trained ML model (.pkl)", type="pkl")

if csv_file and model_file:
    st.write(f"CSV File: `{csv_file.name}`")
    st.write(f"Model File: `{model_file.name}`")

    df = pd.read_csv(csv_file, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)

    try:
        model = joblib.load(model_file)
        st.success("✅ Model and CSV loaded.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    features = ['ema_20', 'ema_50', 'ATR', 'ADX14', 'RSI', 'bb_width',
                'volume_spike_ratio', 'return_1h', 'hour_of_day']
    X = df[features]
    proba = model.predict_proba(X)
    df['predicted_label'] = model.predict(X)
    df['confidence'] = np.max(proba, axis=1)

    if hasattr(model, 'feature_importances_'):
        st.subheader("🧠 Feature Importance")
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values()
        fig_imp, ax_imp = plt.subplots()
        feat_imp.plot(kind='barh', ax=ax_imp)
        st.pyplot(fig_imp)

    threshold = st.slider("🎚 Confidence Threshold", 0.5, 0.9, 0.6, 0.01)
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

    tab1, tab2, tab3, tab4 = st.tabs(["🧾 Signals", "📊 Charts", "📈 Backtest", "📋 Stats"])

    with tab1:
        st.dataframe(df[['close', 'signal', 'confidence']].tail(20))

    with tab2:
        st.subheader("📈 Signal Chart")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df['close'], label='Close Price', color='gray')
        ax.plot(df[df['signal'] == 1].index, df['close'][df['signal'] == 1], '^', color='green', label='Buy')
        ax.plot(df[df['signal'] == -1].index, df['close'][df['signal'] == -1], 'v', color='red', label='Sell')
        ax.legend()
        st.pyplot(fig)

        st.subheader("📈 Price vs Strategy Cumulative Return")
        df['strategy_return'] = df['signal'].shift(1) * df['close'].pct_change()
        df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.plot(df.index, df['close'], label='Price', alpha=0.7)
        ax3b = ax3.twinx()
        ax3b.plot(df.index, df['cumulative_return'], label='Cumulative Return', color='green')
        ax3.legend(loc='upper left'); ax3b.legend(loc='upper right')
        st.pyplot(fig3)

        st.subheader("📊 Signal Count by Day")
        df['date'] = df.index.date
        signal_counts = df.groupby('date')['signal'].value_counts().unstack().fillna(0)
        st.bar_chart(signal_counts)

        st.subheader("🕒 Intraday Signal Heatmap")
        df['hour'] = df.index.hour
        heatmap = df.pivot_table(index=df.index.date, columns='hour', values='signal', aggfunc='sum')
        st.dataframe(heatmap.fillna(0))

    with tab3:
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

            st.download_button(
                label="📥 Download Trades as CSV",
                data=trades_df.to_csv().encode('utf-8'),
                file_name="backtest_trades.csv",
                mime="text/csv"
            )

            trades_df['rolling_win_rate'] = (trades_df['pnl'] > 0).rolling(10).mean()
            st.subheader("📈 Rolling Win Rate")
            st.line_chart(trades_df[['rolling_win_rate']])
        else:
            st.warning("No trades generated.")

    with tab4:
        if not trades_df.empty:
            summary = {
                'Total Legs': len(trades_df),
                'Win Rate (%)': 100 * (trades_df['pnl'] > 0).mean(),
                'Total PnL': trades_df['pnl'].sum(),
                'Avg PnL per Leg': trades_df['pnl'].mean(),
                'Max Drawdown': trades_df['drawdown'].min()
            }
            sharpe = trades_df['pnl'].mean() / trades_df['pnl'].std() * np.sqrt(252) if trades_df['pnl'].std() != 0 else 0
            st.metric("📈 Sharpe Ratio", f"{sharpe:.2f}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total PnL", f"{summary['Total PnL']:.2f}")
            col2.metric("Win Rate", f"{summary['Win Rate (%)']:.1f}%")
            col3.metric("Max Drawdown", f"{summary['Max Drawdown']:.2f}")
            st.dataframe(pd.DataFrame([summary]))
        else:
            st.info("No summary data to display.")
else:
    st.info("📁 Please upload both a CSV and PKL file to begin.")
'''

# Write the updated streamlit_app.py to file
output_path = "/mnt/data/streamlit_app_updated.py"
with open(output_path, "w") as f:
    f.write(updated_streamlit_app_code)

output_path
