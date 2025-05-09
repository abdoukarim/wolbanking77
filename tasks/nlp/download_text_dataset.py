import os, sys
os.environ["KAGGLE_USERNAME"] = input("Enter your Kaggle username: ")
os.environ["KAGGLE_KEY"] = input("Enter your Kaggle key: ")


import kagglehub
import shutil

sys.path.append('.')
from utils.logger import setup_logger

logger = setup_logger("Downloading dataset")
# Download the latest version.
source_dataset_path = kagglehub.dataset_download("abdoukarimkandji/wolbanking77v1")
destination_dataset_path = os.path.join(os.getcwd(), "dataset/text/wolbanking77v1")
shutil.move(source_dataset_path, destination_dataset_path)
logger.info(f"Dataset downloaded and moved to {destination_dataset_path}")
