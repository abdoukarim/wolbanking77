import os, sys
os.environ["KAGGLE_USERNAME"] = input("Enter your Kaggle username: ")
os.environ["KAGGLE_KEY"] = input("Enter your Kaggle key: ")


import kagglehub
import shutil

sys.path.append('.')
from utils.logger import setup_logger

logger = setup_logger("Downloading dataset")
# Download the latest version.
source_dataset_path = kagglehub.dataset_download("abdoukarimkandji/WolBanking77")
logger.info(f"Dataset downloaded to {source_dataset_path}")
