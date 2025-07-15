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

    st.header("🔧 Fee Settings")
    st.caption("Customize fee rates (in %) or flat values")
    intraday_brokerage = st.number_input("Intraday Brokerage (%)", value=0.025) / 100
    delivery_brokerage = st.number_input("Delivery Brokerage (%)", value=0.25) / 100
    stt_rate = st.number_input("STT Rate (%)", value=0.025) / 100
    exchange_rate = st.number_input("Exchange Charges (%)", value=0.00325) / 100
    gst_rate = st.number_input("GST Rate (%)", value=18.0) / 100
    stamp_intraday = st.number_input("Stamp Duty (Intraday, %)", value=0.003) / 100
    stamp_delivery = st.number_input("Stamp Duty (Delivery, %)", value=0.015) / 100
    demat_fee = st.number_input("Demat Charges (Flat ₹)", value=23.60)

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

    with tabs[1]:
        st.subheader("📈 Signal Chart")
        if 'close' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index, df['close'], label='Price', color='gray')
            ax.plot(df[df['signal'] == 1].index, df['close'][df['signal'] == 1], '^', color='green', label='Buy')
            ax.plot(df[df['signal'] == -1].index, df['close'][df['signal'] == -1], 'v', color='red', label='Sell')
            ax.legend()
            st.pyplot(fig)

            df['strategy_return'] = df['signal'].shift(1) * df['close'].pct_change()
            df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
            st.subheader("💹 Cumulative Return vs Price")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(df.index, df['close'], label='Price', alpha=0.7)
            ax2b = ax2.twinx()
            ax2b.plot(df.index, df['cumulative_return'], label='Cumulative Return', color='green')
            st.pyplot(fig2)

    with tabs[2]:
        st.subheader("📊 Backtest Performance")
        trades = run_backtest_simulation(df)
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

            trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
            trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['cumulative_pnl'].cummax()
            trades_df['duration_min'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60

            st.line_chart(trades_df.set_index('exit_time')['cumulative_pnl'])
            st.line_chart(trades_df.set_index('exit_time')['drawdown'])

            fig3, ax3 = plt.subplots()
            ax3.hist(trades_df['duration_min'], bins=30, color='skyblue', edgecolor='black')
            ax3.set_title("Trade Duration (Minutes)")
            st.pyplot(fig3)

    with tabs[4]:
        st.subheader("📉 Model Insights")
        st.write("### Confusion Matrix")
        y_true = df['predicted_label']
        y_pred = model.predict(X)
        cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
        fig4, ax4 = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax4)
        st.pyplot(fig4)

        st.write("### Signal Distribution")
        signal_counts = df['signal'].value_counts().sort_index()
        st.bar_chart(signal_counts)

        st.write("### Confidence Histogram")
        fig5, ax5 = plt.subplots()
        ax5.hist(df['confidence'], bins=30, color='orange', edgecolor='black')
        ax5.set_title("Prediction Confidence Distribution")
        st.pyplot(fig5)

    # Signals and Stats tabs already defined above.
