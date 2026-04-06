import pandas as pd

def create_features(df):
    # daily return
    df['Daily_Return'] = df['Close'].pct_change()

    # rolling mean
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()

    # rolling standard deviation
    df['STD_5'] = df['Close'].rolling(window=5).std()
    df['STD_10'] = df['Close'].rolling(window=10).std()

    # rsi calculation
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

    return df

if __name__ == '__main__':
    # Test block: only runs if this file is executed directly
    df = pd.read_csv('../data/spy_processed.csv', index_col='Date', parse_dates=True)
    df = create_features(df)
    print(df.tail())