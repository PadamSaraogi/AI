import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from backtest import run_backtest_simulation

st.set_page_config(layout="wide")
st.title("📈 Smart Trading Dashboard")

with st.sidebar:
    st.header("Configuration")
    csv_file = st.file_uploader("📂 Upload Indicator CSV", type="csv")
    model_file = st.file_uploader("🧠 Upload Trained Model", type="pkl")

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
        st.subheader("📋 Recent Signals")
        st.dataframe(df[['close', 'signal', 'confidence', 'expected_pnl']].tail(20))

    with tabs[1]:
        st.subheader("📈 Signal Chart (Interactive)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=df[df['signal'] == 1].index, y=df['close'][df['signal'] == 1], mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=df[df['signal'] == -1].index, y=df['close'][df['signal'] == -1], mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down')))
        st.plotly_chart(fig, use_container_width=True)

        df['strategy_return'] = df['signal'].shift(1) * df['close'].pct_change()
        df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
        st.subheader("💹 Cumulative Return vs Price")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(df.index, df['close'], label='Price', alpha=0.7)
        ax2b = ax2.twinx()
        ax2b.plot(df.index, df['cumulative_return'], label='Cumulative Return', color='green')
        st.pyplot(fig2)

    with tabs[2]:
        trades = run_backtest_simulation(df)
        trades_df = pd.DataFrame(trades)

        if not trades_df.empty:
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['cumulative_pnl'].cummax()
            trades_df['duration'] = (pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])).dt.total_seconds() / 60

            st.subheader("📈 Equity Curve")
            fig3, ax3 = plt.subplots(figsize=(12, 3))
            ax3.plot(trades_df['exit_time'], trades_df['cumulative_pnl'], color='blue')
            st.pyplot(fig3)

            st.download_button("📥 Download Trades", trades_df.to_csv().encode(), "trades.csv", "text/csv")

            trades_df['date'] = pd.to_datetime(trades_df['exit_time']).dt.date
            daily_pnl = trades_df.groupby('date')['pnl'].sum().reset_index()
            st.download_button("📆 Download Daily PnL", daily_pnl.to_csv().encode(), "daily_pnl.csv")

    with tabs[3]:
        if not trades_df.empty:
            sharpe = trades_df['pnl'].mean() / trades_df['pnl'].std() * np.sqrt(252) if trades_df['pnl'].std() > 0 else 0
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total PnL", f"{trades_df['pnl'].sum():.2f}")
            col2.metric("Win Rate", f"{(trades_df['pnl'] > 0).mean() * 100:.1f}%")
            col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col4.metric("Trades", f"{len(trades_df)}")
            col5.metric("Avg Hold (min)", f"{trades_df['duration'].mean():.1f}")

    with tabs[4]:
        st.subheader("📊 PnL Histogram")
        fig4, ax4 = plt.subplots()
        trades_df['pnl'].hist(bins=30, ax=ax4)
        st.pyplot(fig4)

        st.subheader("📈 Confidence vs PnL Curve")
        thresholds = np.arange(0.0, 1.01, 0.05)
        pnl_list = []
        for t in thresholds:
            temp_df = df.copy()
            temp_df['signal'] = np.where((temp_df['confidence'] >= t) & (temp_df['expected_pnl'] > 0), temp_df['predicted_label'], 0)
            temp_df['position'] = temp_df['signal'].replace(0, np.nan).ffill()
            trades_tmp = run_backtest_simulation(temp_df)
            pnl_list.append(pd.DataFrame(trades_tmp)['pnl'].sum() if trades_tmp else 0)
        fig5, ax5 = plt.subplots()
        ax5.plot(thresholds, pnl_list, marker='o')
        ax5.set_xlabel("Confidence Threshold")
        ax5.set_ylabel("Total PnL")
        st.pyplot(fig5)

        st.subheader("📊 Confusion Matrix")
        y_true = df['predicted_label']
        y_pred = model.predict(X)
        cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
        fig6, ax6 = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax6)
        st.pyplot(fig6)
else:
    st.info("📁 Please upload both a CSV and PKL file to begin.")
