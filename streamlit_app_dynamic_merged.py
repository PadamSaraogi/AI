import streamlit as st
import pandas as pd
import plotly.express as px

# === Streamlit Configuration ===
st.set_page_config(layout="wide")
st.title("📊 Trading Signal Dashboard")

# === File Upload Section ===
st.sidebar.header("Upload Files")
csv_file = st.sidebar.file_uploader("📂 Upload `5m_signals_enhanced_<STOCK>.csv`", type="csv")

# Check if file is uploaded and not empty
if csv_file is not None:
    try:
        # Check if file content is not empty
        if not csv_file.getvalue().strip():
            st.error("The uploaded file is empty. Please upload a valid file.")
            st.stop()

        # Try reading the CSV with the first column as datetime and setting it as the index
        df_signals = pd.read_csv(csv_file, parse_dates=[0], index_col=0)

        # Check the first few rows to verify the data
        st.write("First few rows of the uploaded dataset:", df_signals.head())

        # Ensure datetime column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df_signals.index):
            st.error("The first column is not being recognized as datetime. Please check your file.")
            st.stop()

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty or does not contain valid data.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while reading the file: {str(e)}")
        st.stop()
else:
    st.warning("Please upload a CSV file to proceed.")

# === Tabs Layout using `st.tabs()` ===
tabs = st.tabs(["Signals", "Backtest", "Performance", "Optimization", "Duration Histogram"])

with tabs[0]:  # Signals Tab
    st.subheader("### Enhanced Signals Data")

    # Filter Data for Time Range Selection (Optional)
    min_date = df_signals.index.min().date()  # Convert to date
    max_date = df_signals.index.max().date()  # Convert to date
    selected_date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Filter the dataframe based on the selected date range
    df_filtered = df_signals[(df_signals.index.date >= selected_date_range[0]) &
                             (df_signals.index.date <= selected_date_range[1])]

    # Display Signal Data Table with Filtering and Sorting
    st.write("### Filtered Signal Data")
    st.dataframe(df_filtered[['predicted_label', 'confidence', 'signal', 'position']])

    # === Signal Heatmap Visualization ===
    st.subheader("### Signal Confidence Heatmap")
    signal_matrix = df_filtered[['predicted_label', 'confidence', 'signal']].pivot_table(
        values='confidence', index='predicted_label', columns='signal', aggfunc='mean'
    )
    fig_heatmap = px.imshow(signal_matrix, text_auto=True, color_continuous_scale='Blues', 
                            title="Signal Confidence Heatmap")
    st.plotly_chart(fig_heatmap)

    # === Signal Distribution Visualization ===
    st.subheader("### Signal Distribution")
    fig_signal_dist = px.histogram(df_filtered, x="signal", color="signal",
                                   title="Signal Distribution (Buy, Sell, Hold)")
    st.plotly_chart(fig_signal_dist)

    # === Signal Count and Summary ===
    total_signals = len(df_filtered)
    buy_signals = df_filtered[df_filtered['signal'] == 1].shape[0]
    sell_signals = df_filtered[df_filtered['signal'] == -1].shape[0]
    hold_signals = df_filtered[df_filtered['signal'] == 0].shape[0]

    st.write(f"Total Signals: {total_signals}")
    st.write(f"Buy Signals: {buy_signals} ({(buy_signals / total_signals) * 100:.2f}%)")
    st.write(f"Sell Signals: {sell_signals} ({(sell_signals / total_signals) * 100:.2f}%)")
    st.write(f"Hold Signals: {hold_signals} ({(hold_signals / total_signals) * 100:.2f}%)")

# Remaining tabs (Backtest, Performance, Optimization, Duration Histogram)
with tabs[1]:  # Backtest Tab
    st.subheader("### Backtest Results")
    # Add your backtest code here

with tabs[2]:  # Performance Tab
    st.subheader("### Performance Metrics")
    # Add performance summary or charts here

with tabs[3]:  # Optimization Tab
    st.subheader("### Optimization Results")
    # Add optimization results here

with tabs[4]:  # Duration Histogram Tab
    st.subheader("### Trade Duration Histogram")
    # Add trade duration histogram here
