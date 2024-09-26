import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Fetch historical price data
def fetch_forex_data(pair, start_date, end_date, interval='1d'):
    data = yf.download(pair, start=start_date, end=end_date, interval=interval)
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    return data

# Calculate moving averages
def calculate_moving_averages(df, short_window=20, long_window=50):
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    return df

# Implement moving average crossover strategy with risk management
def backtest_strategy_with_risk_management(df, capital, max_loss_percent=0.10, position_size_percent=0.05):
    position = 0  # 0: no position, 1: long, -1: short
    buy_price = 0
    total_pnl = 0
    num_trades = 0
    num_winning_trades = 0
    trade_returns = []
    current_capital = capital
    max_loss_threshold = capital * max_loss_percent
    equity_curve = [current_capital]  # To track account balance over time
    trade_log = []  # List to log trades and risk management actions

    for i in range(1, len(df)):
        # Stop trading if losses exceed the max loss threshold
        if current_capital <= capital - max_loss_threshold:
            trade_log.append(f"Backtest stopped at {df.index[i]} due to max loss threshold being exceeded.")
            print("Backtest halted: Max loss threshold exceeded.")
            break

        # Position size (5% of current capital)
        position_size = current_capital * position_size_percent

        # Moving average crossover signals
        if df['Short_MA'].iloc[i] > df['Long_MA'].iloc[i] and position != 1:  # Golden cross (buy)
            if position == -1:  # Closing short position
                pnl = buy_price - df['Close'].iloc[i]
                total_pnl += pnl
                current_capital += pnl
                trade_returns.append(pnl)
                df.at[df.index[i], 'PnL'] = pnl
                trade_log.append(f"Closed short at {df['Close'].iloc[i]} with PnL: {pnl:.2f}")
                print(f"Closed short at {df['Close'].iloc[i]} with profit {pnl:.2f}")

                if pnl > 0:
                    num_winning_trades += 1

            # Go long
            position = 1
            buy_price = df['Close'].iloc[i]
            num_trades += 1
            trade_log.append(f"Bought at {df['Close'].iloc[i]} with position size {position_size:.2f}")
            print(f"Bought at {df['Close'].iloc[i]} with position size {position_size:.2f}")

        elif df['Short_MA'].iloc[i] < df['Long_MA'].iloc[i] and position != -1:  # Death cross (sell/short)
            if position == 1:  # Closing long position
                pnl = df['Close'].iloc[i] - buy_price
                total_pnl += pnl
                current_capital += pnl
                trade_returns.append(pnl)
                df.at[df.index[i], 'PnL'] = pnl
                trade_log.append(f"Closed long at {df['Close'].iloc[i]} with PnL: {pnl:.2f}")
                print(f"Closed long at {df['Close'].iloc[i]} with profit {pnl:.2f}")

                if pnl > 0:
                    num_winning_trades += 1

            # Go short
            position = -1
            buy_price = df['Close'].iloc[i]
            num_trades += 1
            trade_log.append(f"Sold at {df['Close'].iloc[i]} with position size {position_size:.2f}")
            print(f"Sold at {df['Close'].iloc[i]} with position size {position_size:.2f}")

        # Track equity curve
        equity_curve.append(current_capital)

    return total_pnl, num_trades, num_winning_trades, trade_returns, df, trade_log, equity_curve, current_capital

# Sharpe Ratio calculation
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    mean_return = np.mean(returns)
    return_std = np.std(returns)
    if return_std == 0:
        return 0
    sharpe_ratio = (mean_return - risk_free_rate) / return_std
    return sharpe_ratio

# Plot the equity curve
def plot_equity_curve(equity_curve):
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Equity Curve")
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Account Balance")
    plt.grid(True)
    plt.legend()
    plt.show()

# Main function to run the backtest
def run_backtest_with_performance_summary(pair, short_window=20, long_window=50, capital=100000):
    # Define the backtest period
    start = '2022-01-01'
    end = datetime.datetime.today().strftime('%Y-%m-%d')

    # Fetch data
    forex_data = fetch_forex_data(pair, start, end)

    # Calculate moving averages
    forex_data = calculate_moving_averages(forex_data, short_window, long_window)

    # Drop rows with NaN values in moving averages (due to the rolling window)
    forex_data.dropna(subset=['Short_MA', 'Long_MA'], inplace=True)

    # Run the backtest with risk management
    total_pnl, num_trades, num_winning_trades, trade_returns, result_data, trade_log, equity_curve, final_capital = \
        backtest_strategy_with_risk_management(forex_data, capital)

    # Calculate key performance metrics
    winning_percentage = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0
    sharpe_ratio = calculate_sharpe_ratio(trade_returns) if num_trades > 0 else 0

    # Output performance summary
    print(f"\nBacktest Results for {pair}:")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Final Capital: {final_capital:.2f}")
    print(f"Total Trades: {num_trades}")
    print(f"Winning Percentage: {winning_percentage:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Average PnL per trade: {total_pnl/num_trades:.2f}" if num_trades > 0 else "No trades executed")

    # Plot the equity curve
    plot_equity_curve(equity_curve)

    # Save the backtest results with PnL data and trade log
    result_data.to_csv(f'{pair}_backtest_with_risk_management_results.csv')
    
    with open(f'{pair}_trade_log.txt', 'w') as f:
        for log_entry in trade_log:
            f.write(log_entry + '\n')

# Run the backtest on EUR/USD with default settings (20-day and 50-day moving averages)
run_backtest_with_performance_summary('EURUSD=X')
