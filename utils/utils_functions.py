# encoding: utf-8
import os
import pandas as pd
import random
import numpy as np
import torch

from datasets import Dataset
from datasets import concatenate_datasets
from datasets import DatasetDict
from datasets import ClassLabel


def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def load_data(dataset_path, split="full"):
    """
    Load the dataset from the specified path and split it into train and test sets.
    Args:
        dataset_path (str): Path to the dataset directory.
        split (str): Split of the dataset to load. Can be "full", "train", or "test".
    Returns:
        raw_dataset (DatasetDict): A dictionary containing the train and test datasets.
    """
    df_train = pd.read_csv(os.path.join(dataset_path, "{split}/train/train.csv".format(split=split)))
    df_test = pd.read_csv(os.path.join(dataset_path, "{split}/test/test.csv".format(split=split)))
    df_train = df_train[['input_wo', 'label']]
    df_test = df_test[['input_wo', 'label']]
    #removal of punctuation marks
    df_train['input_wo_clean'] = df_train['input_wo'].replace(regex=True, to_replace = '[^\\w\\s]', value = ' ')
    df_test['input_wo_clean'] = df_test['input_wo'].replace(regex=True, to_replace = '[^\\w\\s]', value = ' ')
    # Apply lowercase
    df_train['input_wo_clean'] = df_train['input_wo_clean'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df_test['input_wo_clean'] = df_test['input_wo_clean'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    train_ds = Dataset.from_pandas(df_train[['input_wo_clean', 'label']], split="train")
    test_ds = Dataset.from_pandas(df_test[['input_wo_clean', 'label']], split="test")

    raw_dataset = DatasetDict({"train": train_ds, "test": test_ds})
    raw_dataset =  raw_dataset.rename_column("input_wo_clean", "text") # to match Trainer
    
    return raw_dataset


