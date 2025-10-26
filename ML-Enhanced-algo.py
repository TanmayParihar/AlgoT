"""
ML-ENHANCED TRADING ALGORITHM - Optimized Version
Uses Gradient Boosting, Random Forest, and Neural Networks from scikit-learn
Industry-proven algorithms used by top quant hedge funds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries (all from scikit-learn)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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
        self.scaler = StandardScaler()
        
    def generate_market_data(self, ticker, start_date, end_date):
        """Generate realistic synthetic market data"""
        print(f"🔬 Generating market data for {ticker}...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        np.random.seed(42)
        
        initial_price = 400.0
        mu = 0.0008  # Slightly bullish drift
        sigma = 0.015  # Realistic volatility
        
        returns = np.random.normal(mu, sigma, n_days)
        
        # Add market cycles and trends
        long_cycle = np.sin(np.linspace(0, 4*np.pi, n_days)) * 0.002
        medium_cycle = np.sin(np.linspace(0, 12*np.pi, n_days)) * 0.001
        trend = np.linspace(0, 0.0005, n_days)  # Upward trend
        returns = returns + long_cycle + medium_cycle + trend
        
        # Add momentum (autocorrelation)
        for i in range(1, n_days):
            returns[i] += returns[i-1] * 0.15
        
        price = initial_price * np.exp(np.cumsum(returns))
        
        # OHLC
        high = price * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low = price * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        open_price = np.roll(price, 1)
        open_price[0] = initial_price
        
        # Volume
        base_volume = 50000000
        volume = base_volume * (1 + np.abs(np.random.normal(0, 0.5, n_days)))
        volume = volume * (1 + np.abs(returns) * 10)
        
        data = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': price,
            'Volume': volume.astype(int)
        }, index=dates)
        
        return data
    
    def engineer_features(self, data):
        """Engineer 60+ features for ML"""
        df = data.copy()
        
        print("🔧 Engineering advanced features...")
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            df[f'Close_to_SMA_{period}'] = df['Close'] / df[f'SMA_{period}'] - 1
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        for period in [7, 14, 21, 28]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for period in [10, 20, 30]:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = sma + (std * 2)
            df[f'BB_Lower_{period}'] = sma - (std * 2)
            df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / sma
            df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
        
        # Stochastic
        for period in [14, 21]:
            low_min = df['Low'].rolling(window=period).min()
            high_max = df['High'].rolling(window=period).max()
            df[f'Stochastic_{period}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # Volatility
        for period in [10, 20, 30, 60]:
            df[f'Vol_{period}'] = df['Returns'].rolling(window=period).std()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Momentum
        for period in [5, 10, 20, 30, 60]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # ADX
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr_14 = true_range.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df['ADX'] = dx.rolling(14).mean()
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Return_Lag_{lag}'] = df['Returns'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume_Ratio'].shift(lag)
        
        # Rolling statistics
        for period in [5, 10, 20]:
            df[f'Return_Mean_{period}'] = df['Returns'].rolling(window=period).mean()
            df[f'Return_Std_{period}'] = df['Returns'].rolling(window=period).std()
        
        # Trend indicators
        df['Uptrend'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        df['Strong_Uptrend'] = ((df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200'])).astype(int)
        
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        
        return df
    
    def create_targets(self, df, forward_period=5, threshold=0.015):
        """Create prediction targets"""
        df = df.copy()
        df['Forward_Return'] = df['Close'].shift(-forward_period) / df['Close'] - 1
        
        # Binary classification: Will price go up significantly?
        df['Target'] = (df['Forward_Return'] > threshold).astype(int)
        
        return df
    
    def prepare_data(self, df):
        """Prepare features and targets"""
        # Exclude non-feature columns
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Forward_Return', 
                  'Target', 'Returns', 'Log_Returns', 'EMA_12', 'EMA_26']
        
        feature_cols = [col for col in df.columns if col not in exclude and not df[col].isna().all()]
        
        X = df[feature_cols].copy()
        y = df['Target'].copy()
        
        # Fill NaN
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return X, y, feature_cols
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train ensemble of ML models"""
        print("="*80)
        print("🎓 TRAINING ML MODELS")
        print("="*80)
        print()
        
        # 1. Gradient Boosting (Similar to XGBoost)
        print("🌲 Training Gradient Boosting Classifier...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            verbose=0
        )
        self.gb_model.fit(X_train, y_train)
        
        y_pred = self.gb_model.predict(X_val)
        y_proba = self.gb_model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_proba)
        
        print(f"   ✓ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
        print(f"   ✓ F1: {f1:.4f}, AUC: {auc:.4f}")
        print()
        
        # 2. Random Forest
        print("🌳 Training Random Forest Classifier...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        self.rf_model.fit(X_train, y_train)
        
        y_pred = self.rf_model.predict(X_val)
        y_proba = self.rf_model.predict_proba(X_val)[:, 1]
        
        acc_rf = accuracy_score(y_val, y_pred)
        prec_rf = precision_score(y_val, y_pred, zero_division=0)
        rec_rf = recall_score(y_val, y_pred, zero_division=0)
        f1_rf = f1_score(y_val, y_pred, zero_division=0)
        auc_rf = roc_auc_score(y_val, y_proba)
        
        print(f"   ✓ Accuracy: {acc_rf:.4f}, Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f}")
        print(f"   ✓ F1: {f1_rf:.4f}, AUC: {auc_rf:.4f}")
        print()
        
        # 3. Neural Network
        print("🧠 Training Neural Network (MLP)...")
        # Scale data for neural network
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            random_state=42,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.nn_model.fit(X_train_scaled, y_train)
        
        y_pred = self.nn_model.predict(X_val_scaled)
        y_proba = self.nn_model.predict_proba(X_val_scaled)[:, 1]
        
        acc_nn = accuracy_score(y_val, y_pred)
        prec_nn = precision_score(y_val, y_pred, zero_division=0)
        rec_nn = recall_score(y_val, y_pred, zero_division=0)
        f1_nn = f1_score(y_val, y_pred, zero_division=0)
        auc_nn = roc_auc_score(y_val, y_proba)
        
        print(f"   ✓ Accuracy: {acc_nn:.4f}, Precision: {prec_nn:.4f}, Recall: {rec_nn:.4f}")
        print(f"   ✓ F1: {f1_nn:.4f}, AUC: {auc_nn:.4f}")
        print()
        
        metrics = {
            'GB': {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc},
            'RF': {'acc': acc_rf, 'prec': prec_rf, 'rec': rec_rf, 'f1': f1_rf, 'auc': auc_rf},
            'NN': {'acc': acc_nn, 'prec': prec_nn, 'rec': rec_nn, 'f1': f1_nn, 'auc': auc_nn}
        }
        
        return metrics
    
    def generate_signals(self, X, feature_cols):
        """Generate ensemble trading signals"""
        # Gradient Boosting predictions
        gb_proba = self.gb_model.predict_proba(X[feature_cols])[:, 1]
        
        # Random Forest predictions
        rf_proba = self.rf_model.predict_proba(X[feature_cols])[:, 1]
        
        # Neural Network predictions (needs scaling)
        X_scaled = self.scaler.transform(X[feature_cols])
        nn_proba = self.nn_model.predict_proba(X_scaled)[:, 1]
        
        # Ensemble: Weighted average
        ensemble_score = (
            gb_proba * 0.4 +  # Gradient Boosting gets highest weight
            rf_proba * 0.35 +  # Random Forest
            nn_proba * 0.25    # Neural Network
        )
        
        signals = pd.DataFrame(index=X.index)
        signals['GB_Signal'] = gb_proba
        signals['RF_Signal'] = rf_proba
        signals['NN_Signal'] = nn_proba
        signals['Ensemble_Score'] = ensemble_score
        
        # Generate trading signals (more aggressive thresholds)
        signals['Signal'] = 0
        signals.loc[ensemble_score > 0.55, 'Signal'] = 1   # Buy
        signals.loc[ensemble_score < 0.45, 'Signal'] = -1  # Sell
        
        return signals
    
    def backtest(self, df, signals):
        """Backtest the ML strategy"""
        self.capital = self.initial_capital
        self.trades = []
        self.portfolio_values = []
        shares = 0
        entry_price = 0
        
        for i in range(len(signals)):
            date = signals.index[i]
            price = df.loc[date, 'Close']
            signal = signals['Signal'].iloc[i]
            
            portfolio_value = self.capital + (shares * price)
            self.portfolio_values.append({
                'Date': date,
                'Value': portfolio_value,
                'Price': price,
                'Signal': signal,
                'Position': shares > 0,
                'ML_Score': signals['Ensemble_Score'].iloc[i]
            })
            
            # Buy
            if signal == 1 and shares == 0:
                risk_capital = self.capital * 0.95
                shares = int(risk_capital / price)
                if shares > 0:
                    cost = shares * price
                    self.capital -= cost
                    entry_price = price
                    self.trades.append({
                        'Date': date,
                        'Type': 'BUY',
                        'Price': price,
                        'Shares': shares,
                        'Cost': cost,
                        'ML_Score': signals['Ensemble_Score'].iloc[i]
                    })
            
            # Sell
            elif signal == -1 and shares > 0:
                proceeds = shares * price
                self.capital += proceeds
                profit = proceeds - (shares * entry_price)
                profit_pct = (profit / (shares * entry_price)) * 100
                
                self.trades.append({
                    'Date': date,
                    'Type': 'SELL',
                    'Price': price,
                    'Shares': shares,
                    'Proceeds': proceeds,
                    'Profit': profit,
                    'Profit_Pct': profit_pct,
                    'ML_Score': signals['Ensemble_Score'].iloc[i]
                })
                shares = 0
        
        # Final close
        if shares > 0:
            final_price = df.iloc[-1]['Close']
            proceeds = shares * final_price
            self.capital += proceeds
            profit = proceeds - (shares * entry_price)
            profit_pct = (profit / (shares * entry_price)) * 100
            
            self.trades.append({
                'Date': df.index[-1],
                'Type': 'SELL (Final)',
                'Price': final_price,
                'Shares': shares,
                'Proceeds': proceeds,
                'Profit': profit,
                'Profit_Pct': profit_pct,
                'ML_Score': signals['Ensemble_Score'].iloc[-1]
            })
        
        return pd.DataFrame(self.portfolio_values)
    
    def calculate_metrics(self, portfolio_df):
        """Calculate performance metrics"""
        final_value = portfolio_df['Value'].iloc[-1]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        portfolio_df['Return'] = portfolio_df['Value'].pct_change()
        daily_returns = portfolio_df['Return'].dropna()
        
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        trades_df = pd.DataFrame(self.trades)
        
        if len(trades_df) == 0 or 'Type' not in trades_df.columns:
            return {
                'Initial': self.initial_capital,
                'Final': final_value,
                'Return': total_return,
                'Profit': final_value - self.initial_capital,
                'Trades': 0,
                'WinRate': 0,
                'AvgWin': 0,
                'AvgLoss': 0,
                'Sharpe': sharpe,
                'MaxDD': max_drawdown
            }
        
        sell_trades = trades_df[trades_df['Type'].str.contains('SELL')]
        
        if len(sell_trades) > 0:
            wins = sell_trades['Profit'] > 0
            win_rate = (wins.sum() / len(wins)) * 100
            avg_win = sell_trades[wins]['Profit'].mean() if wins.any() else 0
            avg_loss = sell_trades[~wins]['Profit'].mean() if (~wins).any() else 0
        else:
            win_rate = avg_win = avg_loss = 0
        
        return {
            'Initial': self.initial_capital,
            'Final': final_value,
            'Return': total_return,
            'Profit': final_value - self.initial_capital,
            'Trades': len(sell_trades),
            'WinRate': win_rate,
            'AvgWin': avg_win,
            'AvgLoss': avg_loss,
            'Sharpe': sharpe,
            'MaxDD': max_drawdown
        }
    
    def visualize(self, df, signals, portfolio_df, ticker):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Price with signals
        ax1 = plt.subplot(4, 2, 1)
        ax1.plot(df.index, df['Close'], 'k-', linewidth=1.5, alpha=0.7, label='Price')
        
        buy_idx = signals[signals['Signal'] == 1].index
        sell_idx = signals[signals['Signal'] == -1].index
        
        ax1.scatter(buy_idx, df.loc[buy_idx, 'Close'], 
                   color='green', marker='^', s=200, label='ML Buy', zorder=5, edgecolors='darkgreen', linewidth=2)
        ax1.scatter(sell_idx, df.loc[sell_idx, 'Close'], 
                   color='red', marker='v', s=200, label='ML Sell', zorder=5, edgecolors='darkred', linewidth=2)
        
        ax1.set_title(f'{ticker} - ML Trading Signals', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio value
        ax2 = plt.subplot(4, 2, 2)
        ax2.plot(portfolio_df['Date'], portfolio_df['Value'], 'b-', linewidth=2.5, label='Portfolio')
        ax2.axhline(self.initial_capital, color='r', linestyle='--', linewidth=2, label='Initial')
        ax2.fill_between(portfolio_df['Date'], self.initial_capital, portfolio_df['Value'],
                        where=(portfolio_df['Value'] > self.initial_capital), alpha=0.3, color='green')
        ax2.set_title('Portfolio Growth', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ML Ensemble score
        ax3 = plt.subplot(4, 2, 3)
        ax3.plot(signals.index, signals['Ensemble_Score'], 'purple', linewidth=1.5)
        ax3.axhline(0.55, color='green', linestyle='--', alpha=0.5, label='Buy threshold')
        ax3.axhline(0.45, color='red', linestyle='--', alpha=0.5, label='Sell threshold')
        ax3.fill_between(signals.index, 0.45, 0.55, alpha=0.1, color='gray')
        ax3.set_title('ML Ensemble Confidence', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Individual models
        ax4 = plt.subplot(4, 2, 4)
        ax4.plot(signals.index, signals['GB_Signal'], label='Gradient Boosting', alpha=0.7)
        ax4.plot(signals.index, signals['RF_Signal'], label='Random Forest', alpha=0.7)
        ax4.plot(signals.index, signals['NN_Signal'], label='Neural Network', alpha=0.7)
        ax4.axhline(0.5, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('Individual Model Predictions', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Cumulative returns
        ax5 = plt.subplot(4, 2, 5)
        portfolio_df['CumReturn'] = (1 + portfolio_df['Return'].fillna(0)).cumprod()
        ax5.plot(portfolio_df['Date'], portfolio_df['CumReturn'], 'g-', linewidth=2)
        ax5.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Drawdown
        ax6 = plt.subplot(4, 2, 6)
        portfolio_df['Drawdown'] = (portfolio_df['Value'] / portfolio_df['Value'].cummax() - 1) * 100
        ax6.fill_between(portfolio_df['Date'], portfolio_df['Drawdown'], 0, color='red', alpha=0.3)
        ax6.plot(portfolio_df['Date'], portfolio_df['Drawdown'], 'r-', linewidth=1.5)
        ax6.set_title('Drawdown %', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Returns distribution
        ax7 = plt.subplot(4, 2, 7)
        returns = portfolio_df['Return'].dropna() * 100
        ax7.hist(returns, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax7.axvline(0, color='red', linestyle='--', linewidth=2)
        ax7.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Daily Return %')
        ax7.grid(True, alpha=0.3)
        
        # 8. Trade P&L
        ax8 = plt.subplot(4, 2, 8)
        trades_df = pd.DataFrame(self.trades)
        sell_trades = trades_df[trades_df['Type'].str.contains('SELL')]
        if len(sell_trades) > 0:
            profits = sell_trades['Profit'].values
            colors = ['green' if p > 0 else 'red' for p in profits]
            bars = ax8.bar(range(len(profits)), profits, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax8.axhline(0, color='black', linestyle='-', linewidth=1)
            ax8.set_title('Trade Profit/Loss', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Trade #')
            ax8.set_ylabel('P&L ($)')
            ax8.grid(True, alpha=0.3, axis='y')
            
            for i, (bar, profit) in enumerate(zip(bars, profits)):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'${profit:,.0f}', ha='center', 
                        va='bottom' if profit > 0 else 'top', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/ml_enhanced_results.png', dpi=300, bbox_inches='tight')
        print("📊 Visualization saved!")


def main():
    print("="*80)
    print("🤖 ML-ENHANCED TRADING SYSTEM")
    print("Gradient Boosting + Random Forest + Neural Network Ensemble")
    print("="*80)
    print()
    
    algo = MLTradingSystem(initial_capital=100000)
    
    ticker = 'SPY'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    print(f"📈 Asset: {ticker}")
    print(f"📅 Period: {start_date.date()} to {end_date.date()}")
    print(f"💰 Initial Capital: ${algo.initial_capital:,.2f}")
    print()
    
    # Generate data
    data = algo.generate_market_data(ticker, start_date, end_date)
    print(f"✅ Generated {len(data)} days of data")
    print()
    
    # Engineer features
    df = algo.engineer_features(data)
    print(f"✅ Created {len(df.columns)} features")
    print()
    
    # Create targets
    df = algo.create_targets(df, forward_period=5, threshold=0.015)
    
    # Prepare data
    X, y, feature_cols = algo.prepare_data(df)
    
    # Remove NaN targets
    valid = ~y.isna()
    X = X[valid]
    y = y[valid]
    
    print(f"📊 Dataset: {len(X)} samples, {len(feature_cols)} features")
    print()
    
    # Train/test split
    train_size = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Validation split
    val_size = int(len(X_train) * 0.2)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train = X_train.iloc[:-val_size]
    y_train = y_train.iloc[:-val_size]
    
    # Train models
    metrics = algo.train_models(X_train, y_train, X_val, y_val)
    
    # Generate signals
    print("="*80)
    print("🎯 GENERATING SIGNALS")
    print("="*80)
    print()
    
    test_df = df.loc[X_test.index]
    signals = algo.generate_signals(X_test, feature_cols)
    
    print(f"✅ Signals: Buy={( signals['Signal']==1).sum()}, Sell={(signals['Signal']==-1).sum()}, Hold={(signals['Signal']==0).sum()}")
    print()
    
    # Backtest
    print("="*80)
    print("⚡ BACKTESTING")
    print("="*80)
    print()
    
    portfolio_df = algo.backtest(test_df, signals)
    results = algo.calculate_metrics(portfolio_df)
    
    # Results
    print("="*80)
    print("🏆 RESULTS")
    print("="*80)
    print()
    print(f"💵 Initial:      ${results['Initial']:,.2f}")
    print(f"💰 Final:        ${results['Final']:,.2f}")
    print(f"📈 Profit:       ${results['Profit']:,.2f}")
    print(f"📊 Return:       {results['Return']:.2f}%")
    print()
    print(f"🔄 Trades:       {results['Trades']}")
    print(f"🎯 Win Rate:     {results['WinRate']:.2f}%")
    print(f"💚 Avg Win:      ${results['AvgWin']:,.2f}")
    print(f"💔 Avg Loss:     ${results['AvgLoss']:,.2f}")
    print(f"⚡ Sharpe:       {results['Sharpe']:.2f}")
    print(f"📉 Max DD:       {results['MaxDD']:.2f}%")
    print()
    
    # ML Performance
    print("="*80)
    print("🤖 ML MODEL PERFORMANCE")
    print("="*80)
    print()
    print(f"Gradient Boosting: Acc={metrics['GB']['acc']:.3f}, AUC={metrics['GB']['auc']:.3f}")
    print(f"Random Forest:     Acc={metrics['RF']['acc']:.3f}, AUC={metrics['RF']['auc']:.3f}")
    print(f"Neural Network:    Acc={metrics['NN']['acc']:.3f}, AUC={metrics['NN']['auc']:.3f}")
    print()
    
    # Trades
    print("="*80)
    print("📋 TRADES")
    print("="*80)
    trades_df = pd.DataFrame(algo.trades)
    if len(trades_df) > 0:
        print(trades_df.to_string(index=False))
    print()
    
    # Visualize
    print("🎨 Creating visualizations...")
    algo.visualize(test_df, signals, portfolio_df, ticker)
    
    print()
    print("="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print()
    print("🚀 This ML system combines:")
    print("   ✓ Gradient Boosting (like XGBoost)")
    print("   ✓ Random Forest")
    print("   ✓ Neural Network (MLP)")
    print("   ✓ 60+ engineered features")
    print("   ✓ Ensemble predictions")
    print()


if __name__ == "__main__":
    main()
