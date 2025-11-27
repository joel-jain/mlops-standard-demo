from pathlib import Path
import pandas as pd
import logging

# Architectural Note: We use the logging module so we can track the script's progress
# in the terminal, rather than just using 'print()'.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('Step 1: Downloading Raw Data from UCI Repository...')
    
    # Deep Context: The URL is hardcoded here for reproducibility.
    # In a larger system, this might come from a config file.
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        # We use pandas to read the remote CSV directly
        df = pd.read_csv(url, sep=';')
        
        # Ensure the output directory exists
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the raw file
        df.to_csv(output_path, index=False)
        logger.info(f'Success! Raw data saved to: {output_path}')
        logger.info(f'Data Shape: {df.shape}')
        
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        raise

if __name__ == '__main__':
    # Architectural Justification: We define the path relative to this script
    # so it works on any machine, regardless of where they cloned the repo.
    project_dir = Path(__file__).resolve().parents[3]
    
    # The target path: data/raw/winequality.csv
    raw_data_path = project_dir / 'data' / 'raw' / 'winequality.csv'
    
    main(raw_data_path)
