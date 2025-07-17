import matplotlib.pyplot as plt
import pandas as pd

# Load the grid search results
results_df = pd.read_csv('grid_search_results_BEL.csv')

# Plot the performance metrics
plt.figure(figsize=(10,6))

# Plot win rate vs. ml_thresholds
plt.subplot(2, 2, 1)
plt.plot(results_df['ml_threshold'], results_df['win_rate'], marker='o')
plt.title('Win Rate vs. ML Threshold')
plt.xlabel('ML Threshold')
plt.ylabel('Win Rate (%)')

# Plot total PnL vs. trail multipliers
plt.subplot(2, 2, 2)
plt.plot(results_df['trail_mult'], results_df['total_pnl'], marker='o', color='orange')
plt.title('Total PnL vs. Trail Multiplier')
plt.xlabel('Trail Multiplier')
plt.ylabel('Total PnL')

# Plot max drawdown vs. time limits
plt.subplot(2, 2, 3)
plt.plot(results_df['time_limit'], results_df['max_drawdown'], marker='o', color='red')
plt.title('Max Drawdown vs. Time Limit')
plt.xlabel('Time Limit (minutes)')
plt.ylabel('Max Drawdown')

# Plot trade count vs. adx_target_mult
plt.subplot(2, 2, 4)
plt.plot(results_df['adx_target_mult'], results_df['trade_count'], marker='o', color='green')
plt.title('Trade Count vs. ADX Multiplier')
plt.xlabel('ADX Multiplier')
plt.ylabel('Trade Count')

plt.tight_layout()
plt.show()
