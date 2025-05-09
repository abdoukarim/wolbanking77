import pandas as pd
import numpy as np
import re
from matplotlib import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse

sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import set_seed

set_seed(42)
logger = setup_logger("Training script")

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.log(f"Using {device} device")

# DATASET_PATH = ".dataset/text/wolbanking77v1"

def compute_metrics(y_true, y_pred):
    logger.log("f1_score:", metrics.f1_score(y_true, y_pred, average='weighted'))
    logger.log("precision_score:", metrics.precision_score(y_true, y_pred, average='weighted'))
    logger.log("recall_score:", metrics.recall_score(y_true, y_pred, average='weighted'))

def load_data(dataset_path):
    df_train = pd.read_csv(os.path.join(dataset_path, "/full/train/train.csv"))
    df_test = pd.read_csv(os.path.join(dataset_path, "/full/test/test.csv"))
    df_train = df_train[['input_wo', 'label']]
    df_test = df_test[['input_wo', 'label']]
    #removal of punctuation marks
    df_train['input_wo_clean'] = df_train['input_wo'].replace(regex=True, to_replace = '[^\\w\\s]', value = ' ')
    df_test['input_wo_clean'] = df_test['input_wo'].replace(regex=True, to_replace = '[^\\w\\s]', value = ' ')
    # Apply lowercase
    df_train['input_wo_clean'] = df_train['input_wo_clean'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df_test['input_wo_clean'] = df_test['input_wo_clean'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    X_train = df_train["input_wo_clean"].reset_index(drop=True)
    X_test = df_test["input_wo_clean"].reset_index(drop=True)
    y_train = df_train["label"].reset_index(drop=True)
    y_test = df_test["label"].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(y_train.to_numpy())
    y_train_encoded = le.transform(y_train.to_numpy())
    y_test_encoded = le.transform(y_test.to_numpy())

    num_labels=len(le.classes_)

    return X_train, X_test, y_train_encoded, y_test_encoded, le, num_labels

def apply_bow(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_seq = vectorizer.fit_transform(X_train)
    X_test_seq = vectorizer.transform(X_test)

    return X_train_seq, X_test_seq

def compute_knn(le, X_train_seq, X_test_seq, y_train_encoded, y_test_encoded):
    clf = KNeighborsClassifier()
    clf.fit(X_train_seq, y_train_encoded)
    pred = clf.predict(X_test_seq)
    y_true = [le.classes_[y] for y in y_test_encoded]
    y_pred = [le.classes_[y] for y in pred]

    return compute_metrics(y_true, y_pred)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the text baseline task.",
    )

    # Load the dataset
    X_train, X_test, y_train_encoded, y_test_encoded, le, num_labels = load_data(parser.dataset_dir)
    logger.info("Dataset loaded")

    # Apply Bag of Words
    X_train_seq, X_test_seq = apply_bow(X_train, X_test)
    logger.info("Bag of Words applied")

    # Compute KNN
    compute_knn(le, X_train_seq, X_test_seq, y_train_encoded, y_test_encoded)
    logger.info("KNN computed")

if __name__ == "__main__":
    main()
    logger.info("Training script finished")
