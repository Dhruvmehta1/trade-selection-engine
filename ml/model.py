import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def split_data(df):
    """
    Phase 5: SPLIT
    Performs a 70/30 time-based split on the dataset.
    """
    # Features (X) and Target (y)
    feature_cols = ['Daily_Return', 'SMA_5', 'SMA_10', 'STD_5', 'STD_10', 'RSI_14', 'Signal']
    X = df[feature_cols]
    y = df['Label']
    
    # Step 5.1: Time-Based Split (70% Train, 30% Test)
    split_index = int(len(df) * 0.7)
    
    # Step 5.2: Define Train and Test sets
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    # PHASE 5 CHECKPOINT
    print("\n--- PHASE 5: SPLIT CHECKPOINT ---")
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def train_and_predict(X_train, y_train, X_test, y_test=None):
    """
    Phase 6: MODEL
    Trains a Random Forest and generates predictions using a >0.55 threshold.
    """
    # Step 6.1: Use RandomForestClassifier
    # random_state ensures our results are reproducible
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # Step 6.2: Fit Model (Only on training data!)
    model.fit(X_train, y_train)
    
    # Step 6.3: Predict Probabilities (on test data)
    # predict_proba returns an array with [prob_class_0, prob_class_1]
    # We want the probability of class 1 (a winning trade)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Step 6.4: Decision Rule
    # Take trade if probability > 0.55
    ml_signals = (probs > 0.55).astype(int)
    
    # PHASE 6 CHECKPOINT
    print("\n--- PHASE 6: MODEL CHECKPOINT ---")
    probs_series = pd.Series(probs)
    print("Probability Distribution:")
    print(probs_series.describe())
    
    print("\nFiltered Trades (ML Signals):")
    print(pd.Series(ml_signals).value_counts())
    
    if y_test is not None:
        print("\n--- ML Model Performance Metrics ---")
        print(f"Accuracy:  {accuracy_score(y_test, ml_signals):.4f}")
        print(f"Precision: {precision_score(y_test, ml_signals):.4f}")
        print(f"Recall:    {recall_score(y_test, ml_signals):.4f}")
        print(f"F1 Score:  {f1_score(y_test, ml_signals):.4f}")
        
        cm = confusion_matrix(y_test, ml_signals)
        print("\nConfusion Matrix:")
        print(f"True Negatives (TN):  {cm[0][0]}")
        print(f"False Positives (FP): {cm[0][1]}  <-- Model said BUY, but trade LOST")
        print(f"False Negatives (FN): {cm[1][0]}  <-- Model said SKIP, but trade WON")
        print(f"True Positives (TP):  {cm[1][1]}")

    return probs, ml_signals


if __name__ == "__main__":
    # Test the pipeline end-to-end
    import os
    import sys
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    from data.data_loader import load_data
    from features.feature_engineering import create_features
    from strategies.momentum import momentum_strategy
    from ml.labeling import create_labels
    
    print("Loading and preparing data...")
    df = load_data()
    df = create_features(df)
    df = momentum_strategy(df)
    df = create_labels(df)
    
    # Execute Phase 5
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Execute Phase 6
    probs, ml_signals = train_and_predict(X_train, y_train, X_test, y_test)
