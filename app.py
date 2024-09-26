import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

def fetch_forex_data(pair, start_date, end_date, interval='1d'):
    data = yf.download(pair, start=start_date, end=end_date, interval=interval)
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    return data

def calculate_moving_averages(df, short_window=20, long_window=50):
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    return df

def backtest_strategy_with_risk_management(df, capital, max_loss_percent=0.10, position_size_percent=0.05):
    position = 0
    buy_price = 0
    total_pnl = 0
    num_trades = 0
    num_winning_trades = 0
    trade_returns = []
    current_capital = capital
    max_loss_threshold = capital * max_loss_percent
    equity_curve = [current_capital]
    trade_log = []

    for i in range(1, len(df)):
        if current_capital <= capital - max_loss_threshold:
            trade_log.append(f"Backtest stopped at {df.index[i]} due to max loss threshold being exceeded.")
            print("Backtest halted: Max loss threshold exceeded.")
            break

        position_size = current_capital * position_size_percent

        if df['Short_MA'].iloc[i] > df['Long_MA'].iloc[i] and position != 1:
            if position == -1:
                pnl = buy_price - df['Close'].iloc[i]
                total_pnl += pnl
                current_capital += pnl
                trade_returns.append(pnl)
                df.at[df.index[i], 'PnL'] = pnl
                trade_log.append(f"Closed short at {df['Close'].iloc[i]} with PnL: {pnl:.2f}")
                print(f"Closed short at {df['Close'].iloc[i]} with profit {pnl:.2f}")

                if pnl > 0:
                    num_winning_trades += 1

            position = 1
            buy_price = df['Close'].iloc[i]
            num_trades += 1
            trade_log.append(f"Bought at {df['Close'].iloc[i]} with position size {position_size:.2f}")
            print(f"Bought at {df['Close'].iloc[i]} with position size {position_size:.2f}")

        elif df['Short_MA'].iloc[i] < df['Long_MA'].iloc[i] and position != -1:
            if position == 1:
                pnl = df['Close'].iloc[i] - buy_price
                total_pnl += pnl
                current_capital += pnl
                trade_returns.append(pnl)
                df.at[df.index[i], 'PnL'] = pnl
                trade_log.append(f"Closed long at {df['Close'].iloc[i]} with PnL: {pnl:.2f}")
                print(f"Closed long at {df['Close'].iloc[i]} with profit {pnl:.2f}")

                if pnl > 0:
                    num_winning_trades += 1

            position = -1
            buy_price = df['Close'].iloc[i]
            num_trades += 1
            trade_log.append(f"Sold at {df['Close'].iloc[i]} with position size {position_size:.2f}")
            print(f"Sold at {df['Close'].iloc[i]} with position size {position_size:.2f}")

        equity_curve.append(current_capital)

    return total_pnl, num_trades, num_winning_trades, trade_returns, df, trade_log, equity_curve, current_capital

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    mean_return = np.mean(returns)
    return_std = np.std(returns)
    if return_std == 0:
        return 0
    sharpe_ratio = (mean_return - risk_free_rate) / return_std
    return sharpe_ratio

def plot_equity_curve(equity_curve):
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Equity Curve")
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Account Balance")
    plt.grid(True)
    plt.legend()
    plt.show()

def run_backtest_with_performance_summary(pair, short_window=20, long_window=50, capital=100000):
    start = '2022-01-01'
    end = datetime.datetime.today().strftime('%Y-%m-%d')
    forex_data = fetch_forex_data(pair, start, end)
    forex_data = calculate_moving_averages(forex_data, short_window, long_window)
    forex_data.dropna(subset=['Short_MA', 'Long_MA'], inplace=True)

    total_pnl, num_trades, num_winning_trades, trade_returns, result_data, trade_log, equity_curve, final_capital = \
        backtest_strategy_with_risk_management(forex_data, capital)

    winning_percentage = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0
    sharpe_ratio = calculate_sharpe_ratio(trade_returns) if num_trades > 0 else 0

    print(f"\nBacktest Results for {pair}:")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Final Capital: {final_capital:.2f}")
    print(f"Total Trades: {num_trades}")
    print(f"Winning Percentage: {winning_percentage:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Average PnL per trade: {total_pnl/num_trades:.2f}" if num_trades > 0 else "No trades executed")

    plot_equity_curve(equity_curve)

    result_data.to_csv(f'{pair}_backtest_with_risk_management_results.csv')
    
    with open(f'{pair}_trade_log.txt', 'w') as f:
        for log_entry in trade_log:
            f.write(log_entry + '\n')

run_backtest_with_performance_summary('EURUSD=X')
