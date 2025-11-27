import logging
from pathlib import Path

# Architectural Note: Because we did 'pip install -e .', we can import our 
# internal modules just like standard libraries. This proves the package structure works.
from mlops_standard_demo.data.make_dataset import main as make_data
from mlops_standard_demo.models.train_model import main as train_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [PIPELINE] - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Orchestrates the entire MLOps pipeline:
    1. Ingest Data (Raw)
    2. Process Data (Split)
    3. Train Model (Random Forest)
    """
    project_dir = Path(__file__).resolve().parent
    
    logger.info("🚀 Starting MLOps Pipeline...")
    
    # Step 1: Data Engine
    logger.info("▶️  Phase 1: Running Data Pipeline...")
    make_data(project_dir)
    
    # Step 2: Model Engine
    logger.info("▶️  Phase 2: Running Model Training...")
    train_model(project_dir)
    
    logger.info("✅ Pipeline Complete! The project is consistent and functional.")

if __name__ == "__main__":
    run_pipeline()