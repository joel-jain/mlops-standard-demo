from pathlib import Path
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main(project_dir):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('Step 1: Checking Raw Data...')
    
    # Define paths
    raw_path = project_dir / 'data' / 'raw' / 'winequality.csv'
    processed_dir = project_dir / 'data' / 'processed'
    
    # Create processed folder if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data (Downloading if not exists, for robustness)
    if not raw_path.exists():
        logger.info("Raw data not found. Downloading...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
    else:
        df = pd.read_csv(raw_path)

    logger.info(f"Raw Data Loaded. Shape: {df.shape}")

    # 2. Split Data (The MLOps Standard)
    # Architectural Justification: We fix the random_state=42 to ensure 
    # that our experiments are reproducible. If we change this, the split changes.
    logger.info("Step 2: Splitting into Train and Test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Save to Processed
    train_path = processed_dir / 'train.csv'
    test_path = processed_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"✅ Success! Processed data saved to {processed_dir}")
    logger.info(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

if __name__ == '__main__':
    # Resolve project root directory
    project_dir = Path(__file__).resolve().parents[3]
    
    main(project_dir)