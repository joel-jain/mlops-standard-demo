from pathlib import Path
import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
# Architectural Fix: Import the dedicated RMSE function for modern Scikit-Learn
from sklearn.metrics import root_mean_squared_error

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main(project_dir):
    """
    Trains a Random Forest model on the processed data (../data/processed/train.csv)
    and saves the model artifact to (../models/model.pkl).
    """
    logger.info("Step 1: Loading Processed Data...")
    
    # Define paths
    processed_dir = project_dir / 'data' / 'processed'
    model_dir = project_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Training Data
    train_path = processed_dir / 'train.csv'
    if not train_path.exists():
        logger.error(f"File not found: {train_path}. Run 'make_dataset.py' first!")
        return
        
    train_df = pd.read_csv(train_path)
    
    # Separate Features (X) and Target (y)
    X = train_df.drop('quality', axis=1)
    y = train_df['quality']
    
    logger.info(f"Training on {X.shape[0]} samples with {X.shape[1]} features.")

    # Step 2: Train Model
    logger.info("Step 2: Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Evaluate on training set
    preds = rf.predict(X)
    
    # Code Fix: Use the new dedicated function instead of mean_squared_error(squared=False)
    rmse = root_mean_squared_error(y, preds)
    
    logger.info(f"Model Training Complete. Training RMSE: {rmse:.4f}")

    # Step 3: Serialize Model
    model_path = model_dir / 'model.pkl'
    joblib.dump(rf, model_path)
    logger.info(f"✅ Success! Model serialized and saved to: {model_path}")

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[3]
    main(project_dir)