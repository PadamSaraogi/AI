import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from backtest import run_backtest_simulation  # Must be accessible

st.set_page_config(layout="wide")
st.markdown(
    """
    <div style='display: flex; align-items: center;'><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/512px-React-icon.svg.png' width='60'><h1 style='margin-left: 18px;'>Multi-Stock Portfolio Dashboard</h1></div>
    """, unsafe_allow_html=True
)

# --- Sidebar: Uploads and Parameters ---
st.sidebar.header("Upload Files")
signal_files = st.sidebar.file_uploader(
    "Upload signal_enhanced CSVs (one per stock)", type="csv", accept_multiple_files=True)
grid_files = st.sidebar.file_uploader(
    "Upload grid_search_results CSVs (one per stock)", type="csv", accept_multiple_files=True)
total_portfolio_capital = st.sidebar.number_input("Total Portfolio Capital (₹)", min_value=10000, value=100000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100

def extract_symbol(fname):
    # Adjust filename parser if needed
    return fname.split('_')[-1].split('.')[0].lower()

stock_data = {}
if signal_files and grid_files:
    signal_symbols = [extract_symbol(f.name) for f in signal_files]
    for symbol in signal_symbols:
        sig_file = next((f for f in signal_files if extract_symbol(f.name) == symbol), None)
        grid_file = next((f for f in grid_files if extract_symbol(f.name) == symbol), None)
        if sig_file and grid_file:
            df_signals = pd.read_csv(sig_file, parse_dates=['datetime'])
            df_signals.set_index('datetime', inplace=True)
            df_grid = pd.read_csv(grid_file)
            stock_data[symbol] = {'signals': df_signals, 'grid': df_grid}

symbols_list = list(stock_data.keys())
n_stocks = len(symbols_list)

tabs = st.tabs([
    "Portfolio Overview",
    "Per Symbol Analysis",
    "All Equity Curves",
    "Leaderboard",
    "Optimization",
    "Reporting"
])

# ========== Portfolio Overview Tab ==========
with tabs[0]:
    if n_stocks == 0:
        st.warning("Upload matching pairs for each stock.")
    else:
        # Dynamic portfolio inclusion
        included_symbols = st.multiselect(
            "Include stocks in portfolio calculation:",
            options=symbols_list,
            default=symbols_list,
            format_func=lambda x: x.upper()
        )
        if not included_symbols:
            st.error("Add at least one symbol to the portfolio!")
        else:
            capital_per_stock = total_portfolio_capital // n_stocks
            st.write(f"Allocating ₹{capital_per_stock:,} to each of {n_stocks} stocks.")

            # Backtest and calculations
            all_trades, all_equity_curves = {}, {}
            for symbol in included_symbols:
                df_signals = stock_data[symbol]['signals']
                trades_df, equity_curve = run_backtest_simulation(
                    df_signals,
                    starting_capital=capital_per_stock,
                    risk_per_trade=risk_per_trade
                )
                all_trades[symbol] = trades_df
                all_equity_curves[symbol] = equity_curve

            # Portfolio equity
            portfolio_equity = None
            for eq in all_equity_curves.values():
                portfolio_equity = eq if portfolio_equity is None else portfolio_equity.add(eq, fill_value=0)
            total_portfolio_trades = sum([len(tdf) for tdf in all_trades.values()])
            total_portfolio_netpnl = sum([tdf["net_pnl"].sum() for tdf in all_trades.values()])
            # Combine for drawdown analysis
            if portfolio_equity is not None:
                cumulative_returns = portfolio_equity / portfolio_equity.iloc[0]
                drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
                max_drawdown = drawdowns.min()
                volatility = portfolio_equity.pct_change().std()
                sharpe = portfolio_equity.pct_change().mean() / volatility if volatility > 0 else np.nan
                sortino = portfolio_equity.pct_change().mean() / portfolio_equity.pct_change()[portfolio_equity.pct_change()<0].std() if volatility > 0 else np.nan
            else:
                max_drawdown, volatility, sharpe, sortino = np.nan, np.nan, np.nan, np.nan

            # Portfolio KPIs
            st.markdown("### Portfolio Key Metrics")
            colA, colB, colC, colD, colE = st.columns(5)
            colA.metric("Portfolio Trades", total_portfolio_trades)
            colB.metric("Portfolio Net PnL", f"₹{total_portfolio_netpnl:,.2f}")
            colC.metric("Max Drawdown", f"{max_drawdown:.2%}")
            colD.metric("Sharpe", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
            colE.metric("Sortino", f"{sortino:.2f}" if not np.isnan(sortino) else "N/A")

            # Equity curve + drawdown chart
            with st.expander("Show Detailed Charts"):
                st.subheader("Portfolio Equity Curve")
                fig, ax = plt.subplots(figsize=(10, 4))
                portfolio_equity.plot(ax=ax, color="blue", linewidth=2)
                ax.set_title("Portfolio Equity Curve")
                ax.set_xlabel("Date")
                ax.set_ylabel("Total Value (₹)")
                ax.grid(True)
                st.pyplot(fig)
                st.subheader("Portfolio Max Drawdown")
                fig_dd, ax_dd = plt.subplots(figsize=(10, 3))
                drawdowns.plot(ax=ax_dd, color='red')
                ax_dd.set_title("Portfolio Drawdown")
                ax_dd.set_ylabel("Drawdown (%)")
                ax_dd.grid(True)
                st.pyplot(fig_dd)

            # Portfolio allocation pie chart
            final_vals = [all_trades[s]["capital_after_trade"].iloc[-1] if len(all_trades[s]) else capital_per_stock for s in included_symbols]
            fig2 = go.Figure(data=[go.Pie(
                labels=[s.upper() for s in included_symbols],
                values=final_vals,
                textinfo='label+percent+value',
                hole=0.2
            )])
            fig2.update_layout(title="Portfolio Allocation by Final Capital")
            st.plotly_chart(fig2)

            # Portfolio summary table + download
            summary_data = []
            for symbol in included_symbols:
                trades_df = all_trades[symbol]
                final_capital = trades_df["capital_after_trade"].iloc[-1] if not trades_df.empty else capital_per_stock
                net_pnl = final_capital - capital_per_stock
                win_rate = (trades_df["net_pnl"] > 0).mean() * 100 if not trades_df.empty else 0
                summary_data.append({
                    "Symbol": symbol.upper(),
                    "Start Capital": capital_per_stock,
                    "Final Capital": round(final_capital, 2),
                    "Net PnL": round(net_pnl, 2),
                    "Win Rate (%)": f"{win_rate:.2f}",
                    "Max Drawdown (%)": f"{(drawdowns.min()*100):.2f}" if trades_df is not None else "N/A"
                })
            st.subheader("Portfolio Symbol Summary")
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary)

            csv_summary = df_summary.to_csv(index=False).encode('utf-8')
            st.download_button("Download Portfolio Summary", csv_summary, file_name="portfolio_summary.csv", mime='text/csv')

# ========== Per Symbol Analysis Tab ==========
with tabs[1]:
    if n_stocks == 0:
        st.warning("Upload matching pairs for each stock.")
    else:
        symbol_select = st.selectbox(
            "Choose symbol for per-stock analysis",
            symbols_list, format_func=lambda x: x.upper())
        trades_df = all_trades.get(symbol_select)
        equity_curve = all_equity_curves.get(symbol_select)
        if trades_df is not None:
            # Advanced KPIs in 3 columns
            win_rate = (trades_df["net_pnl"] > 0).mean() * 100 if not trades_df.empty else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", len(trades_df))
            col2.metric("Win Rate (%)", f"{win_rate:.2f}" if not trades_df.empty else "N/A")
            col3.metric("Net PnL", f"₹{trades_df['net_pnl'].sum():,.2f}" if not trades_df.empty else "N/A")
            # Equity curve
            st.markdown("### Equity Curve")
            fig_eq, ax = plt.subplots(figsize=(10,4))
            equity_curve.plot(ax=ax, color="green", linewidth=2)
            ax.set_title(f"{symbol_select.upper()} Equity Curve")
            ax.set_xlabel("Date")
            ax.set_ylabel("Capital (₹)")
            ax.grid(True)
            st.pyplot(fig_eq)
            # Best and Worst trades
            if not trades_df.empty:
                best_trade = trades_df.loc[trades_df['net_pnl'].idxmax()]
                worst_trade = trades_df.loc[trades_df['net_pnl'].idxmin()]
                colb1, colb2 = st.columns(2)
                colb1.success("Best Trade")
                colb1.json(best_trade.to_dict())
                colb2.error("Worst Trade")
                colb2.json(worst_trade.to_dict())
            st.subheader(f"Trades for {symbol_select.upper()}")
            # Date filter for trades
            min_date = trades_df['entry_time'].min()
            max_date = trades_df['exit_time'].max()
            date_range = st.date_input("Filter Trades by Date", [min_date, max_date])
            filtered_trades = trades_df[
                (trades_df['entry_time'] >= pd.to_datetime(date_range[0])) &
                (trades_df['exit_time'] <= pd.to_datetime(date_range[1]))
            ]
            st.dataframe(filtered_trades)
            # Download button for filtered trades
            csv_trade = filtered_trades.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Trades CSV",
                data=csv_trade,
                file_name=f"trades_{symbol_select}.csv",
                mime='text/csv'
            )
            # Win/Loss pie chart
            win_counts = trades_df['net_pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss').value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=win_counts.index,
                values=win_counts.values,
                textinfo='label+percent+value',
                marker=dict(colors=['#4CAF50', '#F44336']),
                hole=0.3,
                pull=[0.1 if label=='Win' else 0 for label in win_counts.index]
            )])
            fig_pie.update_layout(title="Win/Loss Breakdown", showlegend=False)
            st.plotly_chart(fig_pie)

            # Trade signals annotation (for advanced visualization)
            if "signal" in stock_data[symbol_select]['signals']:
                signals_df = stock_data[symbol_select]['signals']
                st.markdown("### Signal Timeline (Buy/Sell/Neutral)")
                signal_colors = signals_df["signal"].map({1:'green', 0:'gray', -1:'red'})
                fig_sig, ax_sig = plt.subplots(figsize=(10,2))
                ax_sig.scatter(signals_df.index, signals_df["signal"], c=signal_colors)
                ax_sig.set_title(f"Signal Timeline: {symbol_select.upper()}")
                ax_sig.set_yticks([-1,0,1])
                ax_sig.set_yticklabels(['Sell','Neutral','Buy'])
                st.pyplot(fig_sig)

# ========== All Equity Curves Tab ==========
with tabs[2]:
    if n_stocks == 0:
        st.warning("Upload matching pairs for each stock.")
    else:
        st.subheader("All Stocks: Equity Curves Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        for symbol, equity_curve in all_equity_curves.items():
            equity_curve.plot(ax=ax, label=symbol.upper())
        ax.set_title("Per-Stock Equity Curves")
        ax.set_xlabel("Date")
        ax.set_ylabel("Capital (₹)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ========== Leaderboard ==========
with tabs[3]:
    if n_stocks == 0:
        st.warning("Upload matching pairs for each stock.")
    else:
        st.markdown("### Stock Leaderboard")
        leaderboard = df_summary.sort_values("Net PnL", ascending=False)
        st.dataframe(leaderboard)
        st.download_button(
            label="Download Leaderboard CSV",
            data=leaderboard.to_csv(index=False).encode('utf-8'),
            file_name="leaderboard.csv",
            mime='text/csv'
        )

# ========== Optimization Results Tab ==========
with tabs[4]:
    st.subheader("Optimization Results: Grid Search")
    if grid_files:
        for symbol in symbols_list:
            opt_df = stock_data[symbol]['grid']
            if not opt_df.empty:
                st.markdown(f"#### {symbol.upper()} Optimization Results")
                st.dataframe(opt_df)
                st.download_button(
                    f"Download {symbol.upper()} Grid Results",
                    opt_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{symbol}_grid_search.csv",
                    mime='text/csv'
                )
                if {"param1", "param2", "win_rate"}.issubset(opt_df.columns):
                    pivot = opt_df.pivot(index="param2", columns="param1", values="win_rate")
                    fig_opt, ax_opt = plt.subplots(figsize=(10, 6))
                    c = ax_opt.pcolormesh(
                        pivot.columns.astype(float), pivot.index.astype(float),
                        pivot.values, cmap="Blues", shading="auto")
                    fig_opt.colorbar(c, ax=ax_opt)
                    ax_opt.set_xlabel("Parameter 1")
                    ax_opt.set_ylabel("Parameter 2")
                    ax_opt.set_title("Grid Search – Win Rate Heatmap")
                    st.pyplot(fig_opt)
    else:
        st.info("Upload grid_search_results CSVs.")

# ========== Reporting Tab ==========
with tabs[5]:
    st.subheader("Exportable Portfolio Report")
    st.markdown(
        """
        Download CSVs for portfolio, per symbol, leaderboard, optimization, or 
        <br><span style='color:gray'>[PDF/HTML reports feature coming soon — ask if you want auto-generated professional reports!]</span>
        """, unsafe_allow_html=True)
    # Place extra download buttons or reporting integrations here

# ========== Theming and Customization ==========
# Streamlit theme can be set in config or via st.markdown(CSS) as needed above.
