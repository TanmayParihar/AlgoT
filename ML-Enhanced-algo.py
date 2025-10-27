"""
ML-ENHANCED DAY TRADING ALGORITHM - Advanced Version with Short Selling
Uses 5-minute timeframe with Gradient Boosting, Random Forest, and Neural Networks
Supports LONG and SHORT positions with sentiment analysis
Multi-instrument and multi-timeframe training to prevent overfitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# For sentiment analysis
import re
from collections import Counter


class SentimentAnalyzer:
    """Lightweight sentiment analyzer for market news"""

    def __init__(self):
        # Sentiment lexicons for financial news
        self.bullish_words = {
            'surge', 'rally', 'soar', 'gain', 'rise', 'climb', 'jump', 'advance', 'upgrade',
            'beat', 'outperform', 'strong', 'growth', 'profit', 'bullish', 'optimistic',
            'positive', 'breakthrough', 'success', 'boom', 'robust', 'solid', 'exceed'
        }

        self.bearish_words = {
            'plunge', 'crash', 'fall', 'drop', 'decline', 'tumble', 'sink', 'slide', 'downgrade',
            'miss', 'underperform', 'weak', 'loss', 'bearish', 'pessimistic', 'negative',
            'failure', 'concern', 'risk', 'worry', 'fear', 'crisis', 'recession'
        }

    def analyze_sentiment(self, text):
        """Analyze sentiment from text, returns score between -1 (bearish) and 1 (bullish)"""
        if not text:
            return 0.0

        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        bullish_count = sum(1 for word in words if word in self.bullish_words)
        bearish_count = sum(1 for word in words if word in self.bearish_words)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        sentiment_score = (bullish_count - bearish_count) / total
        return sentiment_score

    def generate_mock_news(self, date, price_change):
        """Generate mock news headlines based on price movements"""
        if price_change > 0.02:
            headlines = [
                "Markets surge on strong earnings reports",
                "Stocks rally as economic data beats expectations",
                "Bullish sentiment drives market gains"
            ]
        elif price_change < -0.02:
            headlines = [
                "Markets decline on economic concerns",
                "Stocks tumble amid bearish sentiment",
                "Investors worry about market risks"
            ]
        else:
            headlines = [
                "Markets trade mixed in quiet session",
                "Stocks show moderate movement",
                "Market sentiment remains neutral"
            ]

        return np.random.choice(headlines)


class MLTradingSystem:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.portfolio_values = []

        # ML Models
        self.gb_model = None  # Gradient Boosting
        self.rf_model = None  # Random Forest
        self.nn_model = None  # Neural Network
        self.scaler = RobustScaler()  # More robust to outliers

        # Sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()

        # Trading parameters
        self.position_size = 0.30  # Use 30% per trade for day trading
        self.stop_loss = 0.02  # 2% stop loss
        self.take_profit = 0.03  # 3% take profit

    def generate_intraday_data(self, ticker, start_date, end_date, freq='5min'):
        """Generate REALISTIC market-like 5-minute intraday data with complex patterns"""
        print(f"Generating realistic market data for {ticker}...")

        # Seed based on ticker for different behavior per instrument
        seed_map = {'SPY': 42, 'QQQ': 123, 'IWM': 456}
        np.random.seed(seed_map.get(ticker, 42))

        # Starting prices that match real instruments
        price_map = {'SPY': 450.0, 'QQQ': 380.0, 'IWM': 195.0}
        base_price = price_map.get(ticker, 400.0)

        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        all_data = []

        # Volatility state (for GARCH-like clustering)
        volatility_state = 0.015

        for day_idx, date in enumerate(dates):
            # Skip weekends
            if date.weekday() >= 5:
                continue

            # Generate 5-minute bars (9:30 AM - 4:00 PM EST)
            trading_minutes = pd.date_range(
                start=date + timedelta(hours=9, minutes=30),
                end=date + timedelta(hours=16),
                freq='5min'
            )

            # Daily characteristics
            daily_trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # Down, sideways, up
            daily_drift = daily_trend * np.random.uniform(0.0001, 0.0008)

            # Opening gap (common in real markets)
            if day_idx > 0:
                gap = np.random.normal(0, 0.003)  # Overnight gap
                # Larger gaps on Mondays
                if date.weekday() == 0:
                    gap *= 1.5
                base_price *= (1 + gap)

            # Random news events during the day (causes jumps)
            news_times = np.random.choice(len(trading_minutes), size=np.random.randint(0, 3), replace=False)

            prev_return = 0
            for minute_idx, timestamp in enumerate(trading_minutes):
                minute_in_day = minute_idx

                # === Intraday patterns (U-shaped volatility and volume) ===
                time_factor = 1.0
                if minute_in_day < 18:  # First 1.5 hours (high volatility)
                    time_factor = 2.0 - (minute_in_day / 18) * 0.8
                elif minute_in_day > 60:  # Last 1.5 hours (increasing volatility)
                    time_factor = 1.2 + ((minute_in_day - 60) / 18) * 0.8
                else:  # Mid-day (lower volatility)
                    time_factor = 0.7

                # === Volatility clustering (GARCH-like) ===
                # Update volatility state based on recent shocks
                shock = np.random.normal(0, 0.002)
                volatility_state = 0.7 * volatility_state + 0.3 * abs(prev_return) + 0.5 * abs(shock)
                volatility_state = np.clip(volatility_state, 0.005, 0.04)

                current_vol = volatility_state * time_factor

                # === Generate returns with fat tails ===
                # Mix of normal and student-t for fat tails
                if np.random.random() < 0.95:
                    returns = np.random.normal(daily_drift, current_vol)
                else:
                    # Fat tail events (use student-t distribution)
                    returns = np.random.standard_t(df=3) * current_vol * 2

                # === Momentum and mean reversion ===
                momentum = prev_return * 0.15  # Momentum continuation
                mean_reversion = -prev_return * 0.25 if abs(prev_return) > 0.005 else 0  # Mean reversion on large moves
                returns += momentum + mean_reversion

                # === News events (sudden jumps) ===
                if minute_idx in news_times:
                    news_impact = np.random.choice([-1, 1]) * np.random.uniform(0.005, 0.015)
                    returns += news_impact
                    print(f"  News event at {timestamp}: {news_impact*100:.2f}% impact")

                # === Microstructure: bid-ask bounce ===
                bid_ask_bounce = np.random.choice([-1, 1]) * np.random.uniform(0, 0.0005)
                returns += bid_ask_bounce

                # Update price
                base_price *= (1 + returns)

                # === Generate realistic OHLC ===
                # High and low should respect the open-close relationship
                spread = abs(np.random.normal(0, current_vol * 0.5))

                # Determine if bar is bullish or bearish
                is_bullish = returns > 0

                if is_bullish:
                    open_price = all_data[-1]['Close'] if all_data else base_price * 0.999
                    close = base_price
                    high = close * (1 + spread)
                    low = open_price * (1 - spread * 0.6)
                else:
                    open_price = all_data[-1]['Close'] if all_data else base_price * 1.001
                    close = base_price
                    low = close * (1 - spread)
                    high = open_price * (1 + spread * 0.6)

                # Ensure OHLC relationships hold
                high = max(high, open_price, close)
                low = min(low, open_price, close)

                # === Realistic volume patterns ===
                base_volume = 2000000  # Base volume per 5-min bar

                # U-shaped volume (high at open/close)
                volume_factor = 1.0
                if minute_in_day < 12:  # First hour
                    volume_factor = 2.5 - (minute_in_day / 12) * 1.5
                elif minute_in_day > 66:  # Last hour
                    volume_factor = 1.0 + ((minute_in_day - 66) / 12) * 1.5
                else:  # Mid-day
                    volume_factor = 0.6

                # Volume increases with volatility
                vol_multiplier = 1 + abs(returns) * 50

                # Volume spikes on news
                if minute_idx in news_times:
                    vol_multiplier *= 3

                volume = base_volume * volume_factor * vol_multiplier * np.random.uniform(0.8, 1.2)

                all_data.append({
                    'Timestamp': timestamp,
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': int(volume),
                    'Returns': returns
                })

                prev_return = returns

            # End of day: sometimes trend reversal for next day
            if np.random.random() < 0.3:
                daily_trend *= -1

        df = pd.DataFrame(all_data)
        df.set_index('Timestamp', inplace=True)

        print(f"  Generated {len(df)} bars | Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

        return df

    def engineer_features(self, data, ticker='SPY'):
        """Engineer comprehensive features for ML"""
        df = data.copy()

        print(f"Engineering advanced features for {ticker}...")

        # Basic returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Intraday time features
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['TimeOfDay'] = df.index.hour + df.index.minute / 60.0
        df['IsOpen'] = ((df['TimeOfDay'] >= 9.5) & (df['TimeOfDay'] <= 10.5)).astype(int)  # First hour
        df['IsClose'] = ((df['TimeOfDay'] >= 15.0) & (df['TimeOfDay'] <= 16.0)).astype(int)  # Last hour
        df['IsMidDay'] = ((df['TimeOfDay'] >= 11.5) & (df['TimeOfDay'] <= 14.5)).astype(int)  # Lunch period

        # Moving Averages (faster for intraday)
        for period in [3, 6, 12, 24, 48, 78]:  # 15min, 30min, 1hr, 2hr, 4hr, 1day
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            df[f'Close_to_SMA_{period}'] = df['Close'] / df[f'SMA_{period}'] - 1

        # VWAP (Volume Weighted Average Price) - critical for intraday
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['Distance_to_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']

        # Reset VWAP daily
        df['Date'] = df.index.date
        df['VWAP_Daily'] = df.groupby('Date').apply(
            lambda x: (x['Close'] * x['Volume']).cumsum() / x['Volume'].cumsum()
        ).reset_index(level=0, drop=True)

        # RSI (faster periods for intraday)
        for period in [6, 12, 24]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        # MACD (faster for intraday)
        df['EMA_fast'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands
        for period in [12, 24]:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = sma + (std * 2)
            df[f'BB_Lower_{period}'] = sma - (std * 2)
            df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / sma
            df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])

        # Volatility
        for period in [6, 12, 24, 48]:
            df[f'Vol_{period}'] = df['Returns'].rolling(window=period).std()
            df[f'Vol_Change_{period}'] = df[f'Vol_{period}'].pct_change()

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=24).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Change'] = df['Volume'].pct_change()

        # Price momentum
        for period in [3, 6, 12, 24]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100

        # Bid-ask spread proxy (using high-low)
        df['Spread'] = (df['High'] - df['Low']) / df['Close']
        df['Spread_MA'] = df['Spread'].rolling(window=24).mean()

        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f'Return_Lag_{lag}'] = df['Returns'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume_Ratio'].shift(lag)

        # Rolling statistics
        for period in [6, 12, 24]:
            df[f'Return_Mean_{period}'] = df['Returns'].rolling(window=period).mean()
            df[f'Return_Std_{period}'] = df['Returns'].rolling(window=period).std()
            df[f'Return_Skew_{period}'] = df['Returns'].rolling(window=period).skew()
            df[f'Return_Kurt_{period}'] = df['Returns'].rolling(window=period).kurt()

        # Microstructure features
        df['Price_Range'] = (df['High'] - df['Low']) / df['Open']
        df['Body_Size'] = np.abs(df['Close'] - df['Open']) / df['Open']
        df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
        df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']

        # Trend detection
        df['Uptrend_Short'] = (df['SMA_12'] > df['SMA_24']).astype(int)
        df['Uptrend_Long'] = (df['SMA_24'] > df['SMA_78']).astype(int)

        # Add sentiment features (mock for now)
        df['Sentiment'] = df['Returns'].rolling(window=78).apply(
            lambda x: self.sentiment_analyzer.analyze_sentiment(
                self.sentiment_analyzer.generate_mock_news(None, x.iloc[-1])
            )
        )
        df['Sentiment_MA'] = df['Sentiment'].rolling(window=24).mean()
        df['Sentiment_Change'] = df['Sentiment'].diff()

        return df

    def generate_multi_instrument_data(self, tickers, start_date, end_date):
        """Generate and combine data from multiple instruments"""
        print("=" * 80)
        print("GENERATING MULTI-INSTRUMENT DATASET")
        print("=" * 80)

        all_data = []

        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            data = self.generate_intraday_data(ticker, start_date, end_date)
            data = self.engineer_features(data, ticker)
            data['Ticker'] = ticker
            all_data.append(data)
            print(f"  Generated {len(data)} bars for {ticker}")

        combined_df = pd.concat(all_data, axis=0)
        print(f"\nTotal dataset size: {len(combined_df)} bars across {len(tickers)} instruments")

        return combined_df

    def create_targets(self, df, forward_period=12, threshold=0.008):
        """Create THREE-CLASS targets: BUY (1), SELL (-1), HOLD (0)"""
        df = df.copy()

        # Forward returns (handle both single and multi-instrument)
        if 'Ticker' in df.columns:
            df['Forward_Return'] = df.groupby('Ticker')['Close'].shift(-forward_period) / df['Close'] - 1
        else:
            df['Forward_Return'] = df['Close'].shift(-forward_period) / df['Close'] - 1

        # Three-class classification
        df['Target'] = 0  # Hold by default
        df.loc[df['Forward_Return'] > threshold, 'Target'] = 1   # Buy signal
        df.loc[df['Forward_Return'] < -threshold, 'Target'] = -1  # Sell/Short signal

        return df

    def prepare_data(self, df):
        """Prepare features and targets"""
        # Exclude non-feature columns
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Forward_Return',
                  'Target', 'Returns', 'Log_Returns', 'Date', 'Ticker',
                  'EMA_fast', 'EMA_slow', 'VWAP', 'VWAP_Daily']

        feature_cols = [col for col in df.columns if col not in exclude and not df[col].isna().all()]

        X = df[feature_cols].copy()
        y = df['Target'].copy()

        # Fill NaN
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Replace inf
        X = X.replace([np.inf, -np.inf], 0)

        return X, y, feature_cols

    def train_models(self, X_train, y_train, X_val, y_val):
        """Train ensemble of ML models with better regularization"""
        print("=" * 80)
        print("TRAINING ML MODELS")
        print("=" * 80)
        print()

        # Convert three-class to binary for each direction
        # For long signals: 1 vs (0, -1)
        y_train_long = (y_train == 1).astype(int)
        y_val_long = (y_val == 1).astype(int)

        # For short signals: -1 vs (0, 1)
        y_train_short = (y_train == -1).astype(int)
        y_val_short = (y_val == -1).astype(int)

        # 1. Gradient Boosting for LONG
        print("Training Gradient Boosting for LONG signals...")
        self.gb_model_long = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,  # Reduced to prevent overfitting
            learning_rate=0.05,
            subsample=0.7,  # More aggressive subsampling
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            random_state=42,
            verbose=0
        )
        self.gb_model_long.fit(X_train, y_train_long)

        y_pred_long = self.gb_model_long.predict(X_val)
        y_proba_long = self.gb_model_long.predict_proba(X_val)[:, 1]

        print(f"  LONG - Accuracy: {accuracy_score(y_val_long, y_pred_long):.4f}, "
              f"AUC: {roc_auc_score(y_val_long, y_proba_long):.4f}")

        # 2. Gradient Boosting for SHORT
        print("Training Gradient Boosting for SHORT signals...")
        self.gb_model_short = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            random_state=43,  # Different seed
            verbose=0
        )
        self.gb_model_short.fit(X_train, y_train_short)

        y_pred_short = self.gb_model_short.predict(X_val)
        y_proba_short = self.gb_model_short.predict_proba(X_val)[:, 1]

        print(f"  SHORT - Accuracy: {accuracy_score(y_val_short, y_pred_short):.4f}, "
              f"AUC: {roc_auc_score(y_val_short, y_proba_short):.4f}")
        print()

        # 3. Random Forest for LONG
        print("Training Random Forest for LONG signals...")
        self.rf_model_long = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        self.rf_model_long.fit(X_train, y_train_long)

        y_pred_long_rf = self.rf_model_long.predict(X_val)
        y_proba_long_rf = self.rf_model_long.predict_proba(X_val)[:, 1]

        print(f"  LONG - Accuracy: {accuracy_score(y_val_long, y_pred_long_rf):.4f}, "
              f"AUC: {roc_auc_score(y_val_long, y_proba_long_rf):.4f}")

        # 4. Random Forest for SHORT
        print("Training Random Forest for SHORT signals...")
        self.rf_model_short = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=43,
            n_jobs=-1,
            verbose=0
        )
        self.rf_model_short.fit(X_train, y_train_short)

        y_pred_short_rf = self.rf_model_short.predict(X_val)
        y_proba_short_rf = self.rf_model_short.predict_proba(X_val)[:, 1]

        print(f"  SHORT - Accuracy: {accuracy_score(y_val_short, y_pred_short_rf):.4f}, "
              f"AUC: {roc_auc_score(y_val_short, y_proba_short_rf):.4f}")
        print()

        # 5. Neural Network (scaled data)
        print("Training Neural Network for LONG/SHORT signals...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # NN for LONG
        self.nn_model_long = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.01,  # Stronger regularization
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=150,
            random_state=42,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.15
        )
        self.nn_model_long.fit(X_train_scaled, y_train_long)

        y_pred_long_nn = self.nn_model_long.predict(X_val_scaled)
        y_proba_long_nn = self.nn_model_long.predict_proba(X_val_scaled)[:, 1]

        print(f"  NN LONG - Accuracy: {accuracy_score(y_val_long, y_pred_long_nn):.4f}, "
              f"AUC: {roc_auc_score(y_val_long, y_proba_long_nn):.4f}")

        # NN for SHORT
        self.nn_model_short = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=150,
            random_state=43,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.15
        )
        self.nn_model_short.fit(X_train_scaled, y_train_short)

        y_pred_short_nn = self.nn_model_short.predict(X_val_scaled)
        y_proba_short_nn = self.nn_model_short.predict_proba(X_val_scaled)[:, 1]

        print(f"  NN SHORT - Accuracy: {accuracy_score(y_val_short, y_pred_short_nn):.4f}, "
              f"AUC: {roc_auc_score(y_val_short, y_proba_short_nn):.4f}")
        print()

        return {
            'long_acc': accuracy_score(y_val_long, y_pred_long),
            'short_acc': accuracy_score(y_val_short, y_pred_short),
            'long_auc': roc_auc_score(y_val_long, y_proba_long),
            'short_auc': roc_auc_score(y_val_short, y_proba_short)
        }

    def generate_signals(self, X, feature_cols):
        """Generate ensemble trading signals for LONG and SHORT"""
        # LONG predictions
        gb_long = self.gb_model_long.predict_proba(X[feature_cols])[:, 1]
        rf_long = self.rf_model_long.predict_proba(X[feature_cols])[:, 1]

        X_scaled = self.scaler.transform(X[feature_cols])
        nn_long = self.nn_model_long.predict_proba(X_scaled)[:, 1]

        # SHORT predictions
        gb_short = self.gb_model_short.predict_proba(X[feature_cols])[:, 1]
        rf_short = self.rf_model_short.predict_proba(X[feature_cols])[:, 1]
        nn_short = self.nn_model_short.predict_proba(X_scaled)[:, 1]

        # Ensemble scores
        long_score = (gb_long * 0.4 + rf_long * 0.35 + nn_long * 0.25)
        short_score = (gb_short * 0.4 + rf_short * 0.35 + nn_short * 0.25)

        signals = pd.DataFrame(index=X.index)
        signals['Long_Score'] = long_score
        signals['Short_Score'] = short_score

        # Generate signals based on confidence
        signals['Signal'] = 0

        # More aggressive thresholds for day trading
        signals.loc[long_score > 0.60, 'Signal'] = 1   # Buy/Long
        signals.loc[short_score > 0.60, 'Signal'] = -1  # Sell/Short

        # If both are weak, stay neutral
        signals.loc[(long_score < 0.55) & (short_score < 0.55), 'Signal'] = 0

        return signals

    def backtest(self, df, signals):
        """Backtest with LONG and SHORT positions"""
        self.capital = self.initial_capital
        self.trades = []
        self.portfolio_values = []

        position = 0  # 0 = flat, positive = long, negative = short
        entry_price = 0
        position_type = None

        for i in range(len(signals)):
            date = signals.index[i]
            price = df.loc[date, 'Close']
            signal = signals['Signal'].iloc[i]

            current_value = self.capital
            if position != 0:
                if position > 0:  # Long position
                    current_value = self.capital + (position * (price - entry_price))
                else:  # Short position
                    current_value = self.capital + (abs(position) * (entry_price - price))

            self.portfolio_values.append({
                'Date': date,
                'Value': current_value,
                'Price': price,
                'Signal': signal,
                'Position': position,
                'Position_Type': position_type
            })

            # Entry logic
            if signal == 1 and position == 0:  # Enter LONG
                shares = int((self.capital * self.position_size) / price)
                if shares > 0:
                    position = shares
                    entry_price = price
                    position_type = 'LONG'
                    self.trades.append({
                        'Date': date,
                        'Type': 'BUY',
                        'Price': price,
                        'Shares': shares,
                        'Position_Type': position_type
                    })

            elif signal == -1 and position == 0:  # Enter SHORT
                shares = int((self.capital * self.position_size) / price)
                if shares > 0:
                    position = -shares  # Negative for short
                    entry_price = price
                    position_type = 'SHORT'
                    self.trades.append({
                        'Date': date,
                        'Type': 'SHORT',
                        'Price': price,
                        'Shares': shares,
                        'Position_Type': position_type
                    })

            # Exit logic
            if position != 0:
                pnl_pct = 0
                if position > 0:  # Exit LONG
                    pnl_pct = (price - entry_price) / entry_price
                else:  # Exit SHORT
                    pnl_pct = (entry_price - price) / entry_price

                # Stop loss or take profit
                should_exit = False
                if pnl_pct <= -self.stop_loss:
                    should_exit = True
                    exit_reason = 'STOP_LOSS'
                elif pnl_pct >= self.take_profit:
                    should_exit = True
                    exit_reason = 'TAKE_PROFIT'
                elif (position > 0 and signal == -1) or (position < 0 and signal == 1):
                    should_exit = True
                    exit_reason = 'SIGNAL'

                if should_exit:
                    if position > 0:  # Close LONG
                        pnl = position * (price - entry_price)
                        self.capital += pnl
                        self.trades.append({
                            'Date': date,
                            'Type': 'SELL',
                            'Price': price,
                            'Shares': position,
                            'Profit': pnl,
                            'Profit_Pct': pnl_pct * 100,
                            'Exit_Reason': exit_reason
                        })
                    else:  # Close SHORT
                        pnl = abs(position) * (entry_price - price)
                        self.capital += pnl
                        self.trades.append({
                            'Date': date,
                            'Type': 'COVER',
                            'Price': price,
                            'Shares': abs(position),
                            'Profit': pnl,
                            'Profit_Pct': pnl_pct * 100,
                            'Exit_Reason': exit_reason
                        })

                    position = 0
                    entry_price = 0
                    position_type = None

        # Close any remaining position
        if position != 0:
            final_price = df.iloc[-1]['Close']
            if position > 0:
                pnl = position * (final_price - entry_price)
                pnl_pct = (final_price - entry_price) / entry_price
            else:
                pnl = abs(position) * (entry_price - final_price)
                pnl_pct = (entry_price - final_price) / entry_price

            self.capital += pnl
            self.trades.append({
                'Date': df.index[-1],
                'Type': 'CLOSE_FINAL',
                'Price': final_price,
                'Shares': abs(position),
                'Profit': pnl,
                'Profit_Pct': pnl_pct * 100,
                'Exit_Reason': 'END_OF_PERIOD'
            })

        return pd.DataFrame(self.portfolio_values)

    def calculate_metrics(self, portfolio_df):
        """Calculate performance metrics"""
        final_value = portfolio_df['Value'].iloc[-1]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100

        portfolio_df['Return'] = portfolio_df['Value'].pct_change()
        daily_returns = portfolio_df['Return'].dropna()

        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252 * 78)  # Annualized for 5-min
        else:
            sharpe = 0

        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        trades_df = pd.DataFrame(self.trades)

        if len(trades_df) == 0:
            return {
                'Initial': self.initial_capital,
                'Final': final_value,
                'Return': total_return,
                'Profit': final_value - self.initial_capital,
                'Trades': 0,
                'Long_Trades': 0,
                'Short_Trades': 0,
                'WinRate': 0,
                'AvgWin': 0,
                'AvgLoss': 0,
                'Sharpe': sharpe,
                'MaxDD': max_drawdown
            }

        # Separate long and short trades
        completed_trades = trades_df[trades_df['Type'].isin(['SELL', 'COVER', 'CLOSE_FINAL'])]

        long_trades = len(trades_df[trades_df['Type'] == 'BUY'])
        short_trades = len(trades_df[trades_df['Type'] == 'SHORT'])

        if len(completed_trades) > 0 and 'Profit' in completed_trades.columns:
            wins = completed_trades['Profit'] > 0
            win_rate = (wins.sum() / len(wins)) * 100
            avg_win = completed_trades[wins]['Profit'].mean() if wins.any() else 0
            avg_loss = completed_trades[~wins]['Profit'].mean() if (~wins).any() else 0
        else:
            win_rate = avg_win = avg_loss = 0

        return {
            'Initial': self.initial_capital,
            'Final': final_value,
            'Return': total_return,
            'Profit': final_value - self.initial_capital,
            'Trades': len(completed_trades),
            'Long_Trades': long_trades,
            'Short_Trades': short_trades,
            'WinRate': win_rate,
            'AvgWin': avg_win,
            'AvgLoss': avg_loss,
            'Sharpe': sharpe,
            'MaxDD': max_drawdown
        }

    def visualize(self, df, signals, portfolio_df, ticker):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(24, 16))

        # 1. Price with signals
        ax1 = plt.subplot(4, 2, 1)
        ax1.plot(df.index, df['Close'], 'k-', linewidth=1, alpha=0.6, label='Price')

        buy_idx = signals[signals['Signal'] == 1].index
        sell_idx = signals[signals['Signal'] == -1].index

        if len(buy_idx) > 0:
            ax1.scatter(buy_idx, df.loc[buy_idx, 'Close'],
                       color='green', marker='^', s=100, label='LONG', zorder=5, alpha=0.7)
        if len(sell_idx) > 0:
            ax1.scatter(sell_idx, df.loc[sell_idx, 'Close'],
                       color='red', marker='v', s=100, label='SHORT', zorder=5, alpha=0.7)

        ax1.set_title(f'{ticker} - ML Day Trading Signals (5-min)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Portfolio value
        ax2 = plt.subplot(4, 2, 2)
        ax2.plot(portfolio_df['Date'], portfolio_df['Value'], 'b-', linewidth=2, label='Portfolio')
        ax2.axhline(self.initial_capital, color='r', linestyle='--', linewidth=1.5, label='Initial')
        ax2.fill_between(portfolio_df['Date'], self.initial_capital, portfolio_df['Value'],
                        where=(portfolio_df['Value'] > self.initial_capital), alpha=0.3, color='green')
        ax2.fill_between(portfolio_df['Date'], self.initial_capital, portfolio_df['Value'],
                        where=(portfolio_df['Value'] <= self.initial_capital), alpha=0.3, color='red')
        ax2.set_title('Portfolio Value', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Long vs Short scores
        ax3 = plt.subplot(4, 2, 3)
        ax3.plot(signals.index, signals['Long_Score'], 'g-', linewidth=1, alpha=0.7, label='Long Score')
        ax3.plot(signals.index, signals['Short_Score'], 'r-', linewidth=1, alpha=0.7, label='Short Score')
        ax3.axhline(0.60, color='green', linestyle='--', alpha=0.5, label='Long threshold')
        ax3.axhline(0.60, color='red', linestyle='--', alpha=0.5)
        ax3.set_title('ML Confidence: Long vs Short', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Position tracking
        ax4 = plt.subplot(4, 2, 4)
        positions = portfolio_df['Position'].values
        ax4.fill_between(range(len(positions)), 0, positions,
                         where=(positions > 0), alpha=0.3, color='green', label='Long Position')
        ax4.fill_between(range(len(positions)), 0, positions,
                         where=(positions < 0), alpha=0.3, color='red', label='Short Position')
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)
        ax4.set_title('Position Tracking', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Cumulative returns
        ax5 = plt.subplot(4, 2, 5)
        portfolio_df['CumReturn'] = ((portfolio_df['Value'] / self.initial_capital) - 1) * 100
        ax5.plot(portfolio_df['Date'], portfolio_df['CumReturn'], 'b-', linewidth=2)
        ax5.axhline(0, color='r', linestyle='--', linewidth=1)
        ax5.fill_between(portfolio_df['Date'], 0, portfolio_df['CumReturn'],
                        where=(portfolio_df['CumReturn'] > 0), alpha=0.3, color='green')
        ax5.fill_between(portfolio_df['Date'], 0, portfolio_df['CumReturn'],
                        where=(portfolio_df['CumReturn'] <= 0), alpha=0.3, color='red')
        ax5.set_title('Cumulative Return %', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. Drawdown
        ax6 = plt.subplot(4, 2, 6)
        portfolio_df['Drawdown'] = (portfolio_df['Value'] / portfolio_df['Value'].cummax() - 1) * 100
        ax6.fill_between(portfolio_df['Date'], portfolio_df['Drawdown'], 0, color='red', alpha=0.3)
        ax6.plot(portfolio_df['Date'], portfolio_df['Drawdown'], 'r-', linewidth=1.5)
        ax6.set_title('Drawdown %', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # 7. Trade P&L
        ax7 = plt.subplot(4, 2, 7)
        trades_df = pd.DataFrame(self.trades)
        completed = trades_df[trades_df['Type'].isin(['SELL', 'COVER', 'CLOSE_FINAL'])]
        if len(completed) > 0 and 'Profit' in completed.columns:
            profits = completed['Profit'].values
            colors = ['green' if p > 0 else 'red' for p in profits]
            bars = ax7.bar(range(len(profits)), profits, color=colors, alpha=0.7, edgecolor='black')
            ax7.axhline(0, color='black', linestyle='-', linewidth=1)
            ax7.set_title('Trade Profit/Loss', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Trade #')
            ax7.set_ylabel('P&L ($)')
            ax7.grid(True, alpha=0.3, axis='y')

        # 8. Returns distribution
        ax8 = plt.subplot(4, 2, 8)
        if 'Profit_Pct' in completed.columns and len(completed) > 0:
            returns = completed['Profit_Pct'].dropna()
            if len(returns) > 0:
                ax8.hist(returns, bins=30, color='blue', alpha=0.7, edgecolor='black')
                ax8.axvline(0, color='red', linestyle='--', linewidth=2)
                ax8.axvline(returns.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
                ax8.set_title('Trade Returns Distribution', fontsize=14, fontweight='bold')
                ax8.set_xlabel('Return %')
                ax8.legend()
                ax8.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ml_daytrading_results.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to ml_daytrading_results.png")


def main():
    print("=" * 80)
    print("ML-ENHANCED DAY TRADING SYSTEM")
    print("5-Minute Timeframe | Long & Short Positions | Multi-Instrument Training")
    print("=" * 80)
    print()

    algo = MLTradingSystem(initial_capital=100000)

    # Multiple instruments for training
    training_tickers = ['SPY', 'QQQ', 'IWM']  # Different market segments
    test_ticker = 'SPY'

    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # 60 days for intraday

    print(f"Training instruments: {', '.join(training_tickers)}")
    print(f"Test instrument: {test_ticker}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Timeframe: 5-minute bars")
    print(f"Initial Capital: ${algo.initial_capital:,.2f}")
    print()

    # Generate multi-instrument data for training
    print("STEP 1: Generating multi-instrument training data...")
    train_data = algo.generate_multi_instrument_data(training_tickers, start_date, end_date)

    # Create targets
    print("\nSTEP 2: Creating three-class targets (BUY/SELL/HOLD)...")
    train_data = algo.create_targets(train_data, forward_period=12, threshold=0.008)

    # Prepare data
    X, y, feature_cols = algo.prepare_data(train_data)

    # Remove NaN targets
    valid = ~y.isna()
    X = X[valid]
    y = y[valid]

    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Class distribution: BUY={sum(y==1)}, SELL={sum(y==-1)}, HOLD={sum(y==0)}")
    print()

    # Train/val split
    train_size = int(len(X) * 0.75)
    X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

    # Train models
    print("STEP 3: Training ML models...")
    metrics = algo.train_models(X_train, y_train, X_val, y_val)

    # Generate test data (single instrument)
    print("=" * 80)
    print("STEP 4: Generating test data...")
    test_start = end_date - timedelta(days=10)  # Last 10 days for testing
    test_data = algo.generate_intraday_data(test_ticker, test_start, end_date)
    test_data = algo.engineer_features(test_data, test_ticker)
    test_data = algo.create_targets(test_data, forward_period=12, threshold=0.008)

    X_test, y_test, _ = algo.prepare_data(test_data)
    valid_test = ~y_test.isna()
    X_test = X_test[valid_test]
    test_df = test_data.loc[X_test.index]

    print(f"Test data: {len(X_test)} samples")
    print()

    # Generate signals
    print("STEP 5: Generating trading signals...")
    signals = algo.generate_signals(X_test, feature_cols)
    print(f"Signals: LONG={sum(signals['Signal']==1)}, SHORT={sum(signals['Signal']==-1)}, HOLD={sum(signals['Signal']==0)}")
    print()

    # Backtest
    print("=" * 80)
    print("STEP 6: Backtesting strategy...")
    print("=" * 80)
    portfolio_df = algo.backtest(test_df, signals)
    results = algo.calculate_metrics(portfolio_df)

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Initial Capital:    ${results['Initial']:,.2f}")
    print(f"Final Value:        ${results['Final']:,.2f}")
    print(f"Total Profit:       ${results['Profit']:,.2f}")
    print(f"Total Return:       {results['Return']:.2f}%")
    print()
    print(f"Total Trades:       {results['Trades']}")
    print(f"Long Trades:        {results['Long_Trades']}")
    print(f"Short Trades:       {results['Short_Trades']}")
    print(f"Win Rate:           {results['WinRate']:.2f}%")
    print(f"Average Win:        ${results['AvgWin']:,.2f}")
    print(f"Average Loss:       ${results['AvgLoss']:,.2f}")
    print(f"Sharpe Ratio:       {results['Sharpe']:.2f}")
    print(f"Max Drawdown:       {results['MaxDD']:.2f}%")
    print()

    # ML Performance
    print("=" * 80)
    print("ML MODEL PERFORMANCE")
    print("=" * 80)
    print(f"Long Signal Accuracy:  {metrics['long_acc']:.3f}, AUC: {metrics['long_auc']:.3f}")
    print(f"Short Signal Accuracy: {metrics['short_acc']:.3f}, AUC: {metrics['short_auc']:.3f}")
    print()

    # Visualize
    print("STEP 7: Creating visualizations...")
    algo.visualize(test_df, signals, portfolio_df, test_ticker)

    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print()
    print("Enhanced features:")
    print("  5-minute intraday timeframe for day trading")
    print("  LONG and SHORT position support")
    print("  Multi-instrument training (SPY, QQQ, IWM)")
    print("  Sentiment analysis integration")
    print("  Better regularization to prevent overfitting")
    print("  Stop-loss and take-profit management")
    print("  Three-class prediction (BUY/SELL/HOLD)")
    print()


if __name__ == "__main__":
    main()
