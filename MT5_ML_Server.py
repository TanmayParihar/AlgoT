"""
MT5 ML Prediction Server
Provides real-time ML predictions to MetaTrader 5 via socket connection
Uses the trained ML models from the main trading algorithm
"""

import socket
import json
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler


class MT5MLServer:
    def __init__(self, host='127.0.0.1', port=9090):
        self.host = host
        self.port = port
        self.socket = None

        # ML Models (same as Python version)
        self.gb_model_long = None
        self.gb_model_short = None
        self.rf_model_long = None
        self.rf_model_short = None
        self.nn_model_long = None
        self.nn_model_short = None
        self.scaler = RobustScaler()

        self.models_trained = False

    def train_models(self, X_train, y_train):
        """Train ML models (call this once with historical data)"""
        print("Training ML models for MT5...")

        # Convert to binary for long/short
        y_train_long = (y_train == 1).astype(int)
        y_train_short = (y_train == -1).astype(int)

        # Gradient Boosting for LONG
        self.gb_model_long = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            random_state=42
        )
        self.gb_model_long.fit(X_train, y_train_long)

        # Gradient Boosting for SHORT
        self.gb_model_short = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            random_state=43
        )
        self.gb_model_short.fit(X_train, y_train_short)

        # Random Forest for LONG
        self.rf_model_long = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model_long.fit(X_train, y_train_long)

        # Random Forest for SHORT
        self.rf_model_short = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=43,
            n_jobs=-1
        )
        self.rf_model_short.fit(X_train, y_train_short)

        # Neural Network
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.nn_model_long = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            alpha=0.01,
            random_state=42,
            max_iter=150,
            early_stopping=True
        )
        self.nn_model_long.fit(X_train_scaled, y_train_long)

        self.nn_model_short = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            alpha=0.01,
            random_state=43,
            max_iter=150,
            early_stopping=True
        )
        self.nn_model_short.fit(X_train_scaled, y_train_short)

        self.models_trained = True
        print("✓ ML models trained successfully!")

    def save_models(self, filepath='mt5_ml_models.pkl'):
        """Save trained models to file"""
        models = {
            'gb_long': self.gb_model_long,
            'gb_short': self.gb_model_short,
            'rf_long': self.rf_model_long,
            'rf_short': self.rf_model_short,
            'nn_long': self.nn_model_long,
            'nn_short': self.nn_model_short,
            'scaler': self.scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(models, f)
        print(f"✓ Models saved to {filepath}")

    def load_models(self, filepath='mt5_ml_models.pkl'):
        """Load trained models from file"""
        try:
            with open(filepath, 'rb') as f:
                models = pickle.load(f)

            self.gb_model_long = models['gb_long']
            self.gb_model_short = models['gb_short']
            self.rf_model_long = models['rf_long']
            self.rf_model_short = models['rf_short']
            self.nn_model_long = models['nn_long']
            self.nn_model_short = models['nn_short']
            self.scaler = models['scaler']

            self.models_trained = True
            print(f"✓ Models loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"✗ Model file not found: {filepath}")
            return False

    def predict(self, features):
        """Generate ML predictions for given features"""
        if not self.models_trained:
            return {'long_score': 0.5, 'short_score': 0.5, 'error': 'Models not trained'}

        # Ensure features is 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        try:
            # LONG predictions
            gb_long = self.gb_model_long.predict_proba(features)[0, 1]
            rf_long = self.rf_model_long.predict_proba(features)[0, 1]

            X_scaled = self.scaler.transform(features)
            nn_long = self.nn_model_long.predict_proba(X_scaled)[0, 1]

            # SHORT predictions
            gb_short = self.gb_model_short.predict_proba(features)[0, 1]
            rf_short = self.rf_model_short.predict_proba(features)[0, 1]
            nn_short = self.nn_model_short.predict_proba(X_scaled)[0, 1]

            # Ensemble scores
            long_score = (gb_long * 0.4 + rf_long * 0.35 + nn_long * 0.25)
            short_score = (gb_short * 0.4 + rf_short * 0.35 + nn_short * 0.25)

            return {
                'long_score': float(long_score),
                'short_score': float(short_score),
                'timestamp': datetime.now().isoformat(),
                'gb_long': float(gb_long),
                'rf_long': float(rf_long),
                'nn_long': float(nn_long),
                'gb_short': float(gb_short),
                'rf_short': float(rf_short),
                'nn_short': float(nn_short)
            }
        except Exception as e:
            return {'error': str(e)}

    def start_server(self):
        """Start socket server to receive requests from MT5"""
        print(f"Starting ML Server on {self.host}:{self.port}...")

        if not self.models_trained:
            print("WARNING: Models not trained! Load or train models first.")
            return

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)

        print(f"✓ ML Server listening on {self.host}:{self.port}")
        print("Waiting for MT5 connections...")

        try:
            while True:
                client, address = self.socket.accept()
                print(f"\n✓ Connection from {address}")

                try:
                    # Receive data from MT5
                    data = client.recv(4096).decode('utf-8')

                    if not data:
                        continue

                    # Parse request
                    request = json.loads(data)
                    print(f"Received: {request['symbol']} at {request.get('timestamp', 'N/A')}")

                    # Extract features
                    features = np.array(request['features'])

                    # Generate prediction
                    prediction = self.predict(features)

                    # Send response
                    response = json.dumps(prediction)
                    client.send(response.encode('utf-8'))

                    print(f"Sent: LONG={prediction.get('long_score', 0):.3f}, SHORT={prediction.get('short_score', 0):.3f}")

                except json.JSONDecodeError as e:
                    error_response = json.dumps({'error': 'Invalid JSON'})
                    client.send(error_response.encode('utf-8'))
                    print(f"✗ JSON error: {e}")

                except Exception as e:
                    error_response = json.dumps({'error': str(e)})
                    client.send(error_response.encode('utf-8'))
                    print(f"✗ Error: {e}")

                finally:
                    client.close()

        except KeyboardInterrupt:
            print("\n✓ Server stopped by user")
        finally:
            if self.socket:
                self.socket.close()


def create_sample_training_data():
    """Create sample data for initial model training"""
    print("Generating sample training data...")

    # Generate synthetic features (similar to real market data)
    n_samples = 5000
    n_features = 20

    X = np.random.randn(n_samples, n_features)

    # Generate targets with some logic
    y = np.zeros(n_samples)

    for i in range(n_samples):
        # Simple logic: if multiple features positive -> BUY, negative -> SELL
        feature_sum = X[i, :5].sum()

        if feature_sum > 1.0:
            y[i] = 1  # BUY
        elif feature_sum < -1.0:
            y[i] = -1  # SELL
        else:
            y[i] = 0  # HOLD

    return X, y


def main():
    """Main function to demonstrate server usage"""

    server = MT5MLServer(host='127.0.0.1', port=9090)

    # Try to load existing models
    if not server.load_models('mt5_ml_models.pkl'):
        print("\nNo saved models found. Training new models...")

        # Create sample training data
        X_train, y_train = create_sample_training_data()

        # Train models
        server.train_models(X_train, y_train)

        # Save models
        server.save_models('mt5_ml_models.pkl')

    # Start server
    print("\n" + "="*60)
    print("ML PREDICTION SERVER FOR MT5")
    print("="*60)
    server.start_server()


if __name__ == "__main__":
    main()
