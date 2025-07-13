
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from backtest import run_backtest_simulation

st.set_page_config(layout="wide")
st.title("📈 Smart Trading Dashboard")

with st.sidebar:
    st.header("Configuration")
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

    recent_signals = df[df['predicted_label'] != 0].tail(100)
    dynamic_threshold = recent_signals['confidence'].quantile(0.25) if not recent_signals.empty else 0.6

    threshold = st.sidebar.slider("🎚 Confidence Threshold", 0.0, 1.0, float(dynamic_threshold), 0.01)

    df['expected_pnl'] = df['confidence'] * df['predicted_label'] * df['return_1h']
    df['signal'] = np.where((df['confidence'] >= threshold) & (df['expected_pnl'] > 0), df['predicted_label'], 0)
    df['position'] = df['signal'].replace(0, np.nan).ffill()

    tabs = st.tabs(["Signals", "Charts", "Backtest", "Stats", "Insights"])

    with tabs[0]:
        st.subheader("📋 Executed Trades (TP1 + Final Exit Breakdown)")
        trades = run_backtest_simulation(df)

        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

            display_cols = trades_df[[
                'trade_type',
                'entry_time', 'entry_price',
                'tp1_exit_price', 'final_exit_price',
                'pnl_half', 'pnl_final', 'pnl'
            ]].sort_values(by='entry_time', ascending=False).reset_index(drop=True)

            st.dataframe(display_cols.style.format({
                "entry_price": "{:.2f}",
                "tp1_exit_price": "{:.2f}",
                "final_exit_price": "{:.2f}",
                "pnl_half": "{:+.2f}",
                "pnl_final": "{:+.2f}",
                "pnl": "{:+.2f}"
            }))
        else:
            st.info("No trades available yet.")

else:
    st.info("📁 Please upload both a CSV and PKL file to begin.")
