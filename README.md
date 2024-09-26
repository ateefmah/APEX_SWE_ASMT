# How to Use

1. **Install Dependencies**:
   Run the following command to install the required Python libraries:
   ```bash
   pip install pandas yfinance matplotlib numpy
   ```

2. **Run the Backtest**:
   Execute the script to backtest the moving average crossover strategy on the EUR/USD pair:
   ```bash
   python backtest.py
   ```

3. **Customize Settings** (optional):
   - Modify the `short_window`, `long_window`, and `capital` parameters in the `run_backtest_with_performance_summary()` function for different strategies or starting capital.

4. **View Results**:
   - **Performance Summary**: Key metrics (PnL, Sharpe ratio, etc.) are printed in the console.
   - **Equity Curve**: A plot of the account balance over time is displayed.
   - **Log Files**: Trade logs and backtest results are saved as `trade_log.txt` and `results.csv`.
