from data.data_loader import load_data
from features.feature_engineering import create_features
from strategies.momentum import momentum_strategy
from ml.labeling import create_labels

def run_pipeline():
    print("Starting pipeline...\n")

    # Phase 1: Load Data
    print("--- PHASE 1: Loading Data ---")
    df = load_data()
    print(f"Success! Loaded {len(df)} rows.")
    
    # Phase 2: Feature Engineering
    print("\n--- PHASE 2: Creating Features ---")
    df = create_features(df)
    print("Success! Added technical indicators and rolling features.")
    
    # Phase 3: Strategy Signals
    print("\n--- PHASE 3: Generating Momentum Signals ---")
    df = momentum_strategy(df)
    print("Success! Momentum breakout signals applied.")
    
    # Phase 4: Label Generation
    print("\n--- PHASE 4: Generating Labels ---")
    df = create_labels(df)
    print("Success! 5-day future returns and binary labels created.")
    
    # Checkpoint Verification
    print("\n==========================================")
    print("        FINAL PIPELINE CHECKPOINT         ")
    print("==========================================")
    print(df[['Close', 'Signal', 'Future_Return_5d', 'Label']].tail(15))
    
    print("\nSignal Distribution (1 = Buy, -1 = Sell, 0 = Hold):")
    print(df['Signal'].value_counts())
    
    print("\nLabel Distribution (1 = Good Trade, 0 = Bad Trade):")
    print(df['Label'].value_counts())

# Run the whole system
if __name__ == "__main__":
    run_pipeline()