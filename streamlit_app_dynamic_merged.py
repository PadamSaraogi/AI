import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from backtest import run_backtest_simulation

st.set_page_config(layout="wide")
st.title("📈 Smart Trading Dashboard")

# === SIDEBAR CONFIG ===
with st.sidebar:
    st.header("Configuration")
    csv_file = st.file_uploader("📂 Upload your indicator CSV file", type="csv")
    model_file = st.file_uploader("🧠 Upload your trained ML model (.pkl)", type="pkl")

    st.header("🔧 Fee Settings")
    intraday_brokerage = st.number_input("Intraday Brokerage (%)", value=0.025) / 100
    delivery_brokerage = st.number_input("Delivery Brokerage (%)", value=0.25) / 100
    stt_rate = st.number_input("STT Rate (%)", value=0.025) / 100
    exchange_rate = st.number_input("Exchange Charges (%)", value=0.00325) / 100
    gst_rate = st.number_input("GST Rate (%)", value=18.0) / 100
    stamp_intraday = st.number_input("Stamp Duty (Intraday, %)", value=0.003) / 100
    stamp_delivery = st.number_input("Stamp Duty (Delivery, %)", value=0.015) / 100
    demat_fee = st.number_input("Demat Charges (Flat ₹)", value=23.60)

# === MAIN LOGIC ===
if csv_file and model_file:
    df = pd.read_csv(csv_file, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    model = joblib.load(model_file)

    st.success("✅ Model and data loaded.")
    features = ['ema_20', 'ema_50', 'ATR', 'ADX14', 'RSI', 'bb_width',
                'volume_spike_ratio', 'return_1h', 'hour_of_day']
    X = df[features]
    proba = model.predict_proba(X)
    df['predicted_label'] = model.predict(X)
    df['confidence'] = np.max(proba, axis=1)

    if hasattr(model, 'feature_importances_'):
        st.subheader("🧠 Feature Importance")
        st.bar_chart(pd.Series(model.feature_importances_, index=X.columns).sort_values())

    recent_signals = df[df['predicted_label'] != 0].tail(100)
    dynamic_threshold = recent_signals['confidence'].quantile(0.25) if not recent_signals.empty else 0.6
    threshold = st.sidebar.slider("🎚 Confidence Threshold", 0.0, 1.0, float(dynamic_threshold), 0.01)

    df['expected_pnl'] = df['confidence'] * df['predicted_label'] * df['return_1h']
    df['signal'] = np.where((df['confidence'] >= threshold) & (df['expected_pnl'] > 0),
                            df['predicted_label'], 0)
    df['position'] = df['signal'].replace(0, np.nan).ffill()

    # === BACKTEST ===
    trades = run_backtest_simulation(df)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    if not trades_df.empty:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df['duration_min'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
        trades_df['tp1_hit'] = trades_df['tp1_exit_price'].notna() & (trades_df['tp1_exit_price'] != 0)
        trades_df['pnl'] = np.where(
            trades_df['tp1_hit'],
            0.5 * (trades_df['tp1_exit_price'] - trades_df['entry_price']) +
            0.5 * (trades_df['final_exit_price'] - trades_df['entry_price']),
            trades_df['final_exit_price'] - trades_df['entry_price']
        )
        trades_df['day_trade'] = trades_df['entry_time'].dt.date == trades_df['exit_time'].dt.date
        buy_val = trades_df['entry_price']
        sell_val = trades_df['final_exit_price']
        total_val = buy_val + sell_val

        trades_df['brokerage'] = np.where(trades_df['day_trade'],
                                          intraday_brokerage * total_val,
                                          delivery_brokerage * total_val)
        trades_df['stt'] = np.where(trades_df['day_trade'],
                                    stt_rate * sell_val,
                                    stt_rate * total_val)
        trades_df['exchange'] = exchange_rate * total_val
        trades_df['gst'] = gst_rate * (trades_df['brokerage'] + trades_df['exchange'])
        trades_df['stamp'] = np.where(trades_df['day_trade'],
                                      stamp_intraday * buy_val,
                                      stamp_delivery * buy_val)
        trades_df['demat'] = np.where(trades_df['day_trade'], 0.0, demat_fee)
        trades_df['fees'] = trades_df[['brokerage', 'stt', 'exchange', 'gst', 'stamp', 'demat']].sum(axis=1)
        trades_df['net_pnl'] = trades_df['pnl'] - trades_df['fees']

        # ✅ SAFELY map confidence to each trade using merge_asof
        df_reset = df.reset_index()[['datetime', 'confidence']].sort_values('datetime')
        trades_df = trades_df.sort_values('entry_time')
        trades_df = pd.merge_asof(trades_df, df_reset, left_on='entry_time', right_on='datetime', direction='nearest')

    # === TABS ===
    tabs = st.tabs(["Trades", "Charts", "Backtest", "Sensitivity", "Insights"])

    with tabs[0]:
        st.subheader("📊 Trade Summary Stats")

    if not trades_df.empty:
        # --- Overall Summary ---
        total_trades = len(trades_df)
        profitable = (trades_df['net_pnl'] > 0).sum()
        win_rate = profitable / total_trades * 100
        avg_duration = trades_df['duration_min'].mean()
        total_fees = trades_df['fees'].sum()
        gross_pnl = trades_df['pnl'].sum()
        net_pnl = trades_df['net_pnl'].sum()

        st.markdown("### 📌 Overall Summary")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Avg Duration", f"{avg_duration:.1f} min")
        col4.metric("Gross PnL", f"{gross_pnl:.2f}")
        col5.metric("Net PnL", f"{net_pnl:.2f}")
        col6.metric("Total Fees", f"{total_fees:.2f}")

        # --- Intraday Summary ---
        intraday_df = trades_df[trades_df['day_trade']]
        st.markdown("### 💼 Intraday Trades")
        if not intraday_df.empty:
            intraday_winrate = (intraday_df['net_pnl'] > 0).mean() * 100
        else:
            intraday_winrate = 0.0
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Trades", len(intraday_df))
        col2.metric("Win Rate", f"{intraday_winrate:.1f}%")
        col3.metric("Avg Duration", f"{intraday_df['duration_min'].mean():.1f} min" if not intraday_df.empty else "0.0 min")
        col4.metric("Gross PnL", f"{intraday_df['pnl'].sum():.2f}")
        col5.metric("Net PnL", f"{intraday_df['net_pnl'].sum():.2f}")
        col6.metric("Fees", f"{intraday_df['fees'].sum():.2f}")

        # --- Delivery Summary ---
        delivery_df = trades_df[~trades_df['day_trade']]
        st.markdown("### 📦 Delivery Trades")
        if not delivery_df.empty:
            delivery_winrate = (delivery_df['net_pnl'] > 0).mean() * 100
        else:
            delivery_winrate = 0.0
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Trades", len(delivery_df))
        col2.metric("Win Rate", f"{delivery_winrate:.1f}%")
        col3.metric("Avg Duration", f"{delivery_df['duration_min'].mean():.1f} min" if not delivery_df.empty else "0.0 min")
        col4.metric("Gross PnL", f"{delivery_df['pnl'].sum():.2f}")
        col5.metric("Net PnL", f"{delivery_df['net_pnl'].sum():.2f}")
        col6.metric("Fees", f"{delivery_df['fees'].sum():.2f}")
        # --- Filters ---
        st.subheader("🧰 Trade Filters")
        with st.expander("🔍 Filter Options"):
            trade_type_filter = st.selectbox("Trade Type", options=["All", "Buy", "Sell"])
            daytrade_only = st.checkbox("Show only Intraday Trades", value=True)
            daytrade_only = st.checkbox("Show only Delivery Trades", value=False)

        filtered_df = trades_df.copy()
        if trade_type_filter != "All":
            filtered_df = filtered_df[filtered_df['trade_type'] == trade_type_filter]
        if daytrade_only:
            filtered_df = filtered_df[filtered_df['day_trade']]

        # --- Breakdown Table ---
        st.subheader("📋 Executed Trades – Detailed Breakdown")

        filtered_df['pnl_half'] = np.where(filtered_df['tp1_hit'],
                                           0.5 * (filtered_df['tp1_exit_price'] - filtered_df['entry_price']),
                                           0.0)
        filtered_df['pnl_final'] = np.where(filtered_df['tp1_hit'],
                                            0.5 * (filtered_df['final_exit_price'] - filtered_df['entry_price']),
                                            filtered_df['final_exit_price'] - filtered_df['entry_price'])

        display_cols = filtered_df[[
            'entry_time', 'exit_time', 'entry_price', 'tp1_exit_price', 'final_exit_price',
            'pnl_half', 'pnl_final', 'pnl', 'fees', 'net_pnl',
            'brokerage', 'stt', 'exchange', 'gst', 'stamp', 'demat',
            'trade_type', 'day_trade', 'duration_min'
        ]]

        st.dataframe(display_cols.style.format({
            'entry_price': '{:.2f}', 'tp1_exit_price': '{:.2f}', 'final_exit_price': '{:.2f}',
            'pnl_half': '{:+.2f}', 'pnl_final': '{:+.2f}', 'pnl': '{:+.2f}',
            'fees': '{:.2f}', 'net_pnl': '{:+.2f}', 'duration_min': '{:.1f}',
            'brokerage': '{:.2f}', 'stt': '{:.2f}', 'exchange': '{:.2f}',
            'gst': '{:.2f}', 'stamp': '{:.2f}', 'demat': '{:.2f}'
        }))


    with tabs[1]:
        st.subheader("📈 Price with Signal Overlay")
        if 'close' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df['close'], label='Price', color='gray')
        ax.plot(df[df['signal'] == 1].index, df['close'][df['signal'] == 1], '^', color='green', label='Buy')
        ax.plot(df[df['signal'] == -1].index, df['close'][df['signal'] == -1], 'v', color='red', label='Sell')
        ax.legend()
        st.pyplot(fig)
    
        st.subheader("📊 Cumulative Return %: Strategy vs Buy & Hold")
    
        # Calculate strategy and buy-hold cumulative returns
        df['strategy_return'] = df['signal'].shift(1) * df['close'].pct_change()
        df['cumulative_strategy'] = (1 + df['strategy_return'].fillna(0)).cumprod()
        df['cumulative_hold'] = df['close'] / df['close'].iloc[0]
    
        # Convert to percentage
        df['cumulative_strategy_pct'] = (df['cumulative_strategy'] - 1) * 100
        df['cumulative_hold_pct'] = (df['cumulative_hold'] - 1) * 100
    
        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df['cumulative_strategy_pct'], label='Strategy', color='green')
        ax.plot(df.index, df['cumulative_hold_pct'], label='Buy & Hold', color='blue')
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_title("Cumulative Return %: Strategy vs Buy & Hold")
        ax.legend()
        st.pyplot(fig)
    
    
        # Volatility Chart (ATR)
        st.subheader("🌪 ATR (Volatility)")
        if 'ATR' in df.columns:
            st.line_chart(df['ATR'])
    
        # Drawdown Curve
        st.subheader("📉 Strategy Drawdown")
        drawdown = df['cumulative_strategy'] - df['cumulative_strategy'].cummax()
        st.line_chart(drawdown)
    
        # Signal Frequency by Hour
        st.subheader("🕒 Signal Frequency by Hour")
        if 'signal' in df.columns:
            df_signals = df[df['signal'] != 0].copy()
        df_signals['hour'] = df_signals.index.hour
        hourly_signals = df_signals.groupby('hour')['signal'].count()
        st.bar_chart(hourly_signals)
    
        # Rolling Sharpe Ratio (Advanced)
        st.subheader("📐 Rolling Sharpe Ratio (30-period)")
        rolling_sharpe = df['strategy_return'].rolling(30).mean() / df['strategy_return'].rolling(30).std()
        st.line_chart(rolling_sharpe)
    

    with tabs[2]:
        st.subheader("📊 Backtest Performance")
        if not trades_df.empty:
            trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
            trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['cumulative_pnl'].cummax()
            st.line_chart(trades_df.set_index('exit_time')['cumulative_pnl'])
            st.line_chart(trades_df.set_index('exit_time')['drawdown'])

    with tabs[3]:
        st.subheader("🎛 Threshold Sensitivity")
        if not trades_df.empty:
            thresholds = np.arange(0.4, 0.91, 0.05)
            results = []
            for t in thresholds:
                sub = df[(df['confidence'] >= t) & (df['expected_pnl'] > 0)]
                signal_count = len(sub)
                wins = (sub['predicted_label'] == sub['signal']).sum()
                avg_conf = sub['confidence'].mean() if not sub.empty else 0
                sub_trades = trades_df[trades_df['confidence'] >= t]
                gross_pnl = sub_trades['pnl'].sum() if not sub_trades.empty else 0
                results.append((t, signal_count, wins, avg_conf, gross_pnl))

            sens_df = pd.DataFrame(results, columns=["Threshold", "Signal Count", "Predicted Match", "Avg Confidence", "Gross PnL"])
            st.dataframe(sens_df.set_index("Threshold"))

    with tabs[4]:
        st.subheader("📉 Model Insights")
        y_true = df['predicted_label']
        y_pred = model.predict(X)
        cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax)
        st.pyplot(fig)
