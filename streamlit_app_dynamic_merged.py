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

    with tabs[0]:
        st.subheader("📋 Executed Trades (TP1 + Final Exit Breakdown)")
        trades = run_backtest_simulation(df)

        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

            has_tp1 = trades_df['tp1_exit_price'].notna() & (trades_df['tp1_exit_price'] != 0)
            trades_df['tp1_hit'] = has_tp1
            trades_df['pnl'] = np.where(
                has_tp1,
                0.5 * (trades_df['tp1_exit_price'] - trades_df['entry_price']) + 0.5 * (trades_df['final_exit_price'] - trades_df['entry_price']),
                trades_df['final_exit_price'] - trades_df['entry_price']
            )

            entry_val = trades_df['entry_price']
            exit_val = trades_df['final_exit_price']
            trades_df['day_trade'] = trades_df['entry_time'].dt.date == trades_df['exit_time'].dt.date

            buy_val = entry_val
            sell_val = exit_val
            total_val = buy_val + sell_val

            trades_df['brokerage'] = np.where(trades_df['day_trade'], intraday_brokerage * total_val, delivery_brokerage * total_val)
            trades_df['stt'] = np.where(trades_df['day_trade'], stt_rate * sell_val, stt_rate * total_val)
            trades_df['exchange'] = exchange_rate * total_val
            trades_df['gst'] = gst_rate * (trades_df['brokerage'] + trades_df['exchange'])
            trades_df['stamp'] = np.where(trades_df['day_trade'], stamp_intraday * buy_val, stamp_delivery * buy_val)
            trades_df['demat'] = np.where(trades_df['day_trade'], 0.0, demat_fee)

            trades_df['fees'] = trades_df[['brokerage', 'stt', 'exchange', 'gst', 'stamp', 'demat']].sum(axis=1)
            trades_df['net_pnl'] = trades_df['pnl'] - trades_df['fees']

            trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
            monthly_pnl = trades_df.groupby('month')['net_pnl'].sum()
            monthly_count = trades_df.groupby('month').size()

            display_cols = trades_df[[
                'trade_type', 'entry_time', 'entry_price', 'tp1_exit_price',
                'final_exit_price', 'tp1_hit', 'pnl', 'fees', 'net_pnl']].sort_values(by='entry_time', ascending=False).reset_index(drop=True)

            st.dataframe(display_cols.style.format({
                "entry_price": "{:.2f}",
                "tp1_exit_price": "{:.2f}",
                "final_exit_price": "{:.2f}",
                "pnl": "{:+.2f}",
                "fees": "{:.2f}",
                "net_pnl": "{:+.2f}"
            }))

            st.subheader("📅 Monthly Net PnL Breakdown")
            st.bar_chart(monthly_pnl)

            st.subheader("📊 Monthly Trade Count")
            st.bar_chart(monthly_count)

            st.subheader("📌 TP1 Hit vs Not Hit Summary")
            tp1_summary = trades_df.groupby('tp1_hit').agg(
                net_pnl=('net_pnl', 'sum'),
                avg_fee=('fees', 'mean'),
                count=('net_pnl', 'count')
            ).rename(index={True: 'TP1 Hit', False: 'No TP1'})
            st.dataframe(tp1_summary.style.format({
                'net_pnl': '{:+.2f}',
                'avg_fee': '{:.2f}',
                'count': '{:d}'
            }))

            st.subheader("📊 Fee Component Breakdown (Last Trade)")
            if not trades_df.empty:
                last = trades_df.iloc[-1]
                fee_labels = ['Brokerage', 'STT', 'Exchange', 'GST', 'Stamp', 'Demat']
                fee_values = [last.brokerage, last.stt, last.exchange, last.gst, last.stamp, last.demat]
                fig, ax = plt.subplots()
                ax.pie(fee_values, labels=fee_labels, autopct='%1.2f%%', startangle=90)
                ax.set_title("Fee Breakdown")
                st.pyplot(fig)

    with tabs[3]:
        st.subheader("📈 Stats")
        if not trades_df.empty:
            sharpe = trades_df['net_pnl'].mean() / trades_df['net_pnl'].std() * np.sqrt(252) if trades_df['net_pnl'].std() > 0 else 0
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Gross PnL", f"{trades_df['pnl'].sum():.2f}")
            col2.metric("Net PnL", f"{trades_df['net_pnl'].sum():.2f}")
            col3.metric("Total Fees", f"{trades_df['fees'].sum():.2f}")
            col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col5.metric("Trades", f"{len(trades_df)}")
else:
    st.info("📁 Please upload both a CSV and PKL file to begin.")
