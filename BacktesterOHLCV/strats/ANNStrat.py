# ann_strat.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from engine.strategyClass import OHLCVTradingStrategy

class ANNStrat(OHLCVTradingStrategy):
    def __init__(self, name, features, hidden_units=[32, 16], epochs=5, batch_size=64):
        super().__init__(name)
        self.features = features  # feature column names
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.prediction_map = {}
        self.history = None  # store training history

    def load_data(self, df):
        df = df.dropna(subset=self.features + ['signal'])
        self.train_df = df.copy()

    def train_model(self, validation_split=0.5, verbose=1):
        df = self.train_df.copy()
        X = df[self.features].values
        y = df['signal'].values + 1  # shift -1, 0, 1 -> 0,1,2

        # Split
        split_idx = int((1 - validation_split) * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Compute class weights
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = {i: w for i, w in zip(classes, weights)}

        # Build model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(len(self.features),)))
        for units in self.hidden_units:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(3, activation='softmax'))  # 3 classes

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Add early stopping
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        ]

        # Train
        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose
        )

        self.model = model

        # Predict on full dataset
        preds = model.predict(X, verbose=0)
        predicted_classes = np.argmax(preds, axis=1) - 1  # back to -1,0,1

        for (date, ticker, pred) in zip(df['date'], df['ticker'], predicted_classes):
            self.prediction_map[(ticker, pd.to_datetime(date))] = pred

    def generate_signals(self, asset_df):
        ticker = asset_df['ticker'].iloc[0] if 'ticker' in asset_df.columns else None
        dates = asset_df['date'] if 'date' in asset_df.columns else asset_df.index
        signals = []

        for date in dates:
            key = (ticker, pd.to_datetime(date))
            signals.append(self.prediction_map.get(key, 0))  # default to hold

        indicators = [None] * len(signals)
        return signals, indicators
