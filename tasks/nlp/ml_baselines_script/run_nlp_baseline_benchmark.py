import pandas as pd
import numpy as np
import re
# from matplotlib import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
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
logger.info(f"Using {device} device")

# DATASET_PATH = ".dataset/text/wolbanking77v1"

def export_results_to_csv(precision, recall, f1, output_dir, split):
    data = {
        'Model': ['BoW+KNN', 'BoW+SVM', 'BoW+LR', 'BoW+NB'],
        'split': ['full', 'full', 'full', 'full'],
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

    results_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    results_df.to_csv(os.path.join(output_dir, "benchmark_baseline_results_{}.csv".format(split=split)), index=False)


def compute_metrics(y_true, y_pred):
    logger.info("f1_score:")
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    logger.info(f1)
    logger.info("precision_score:")
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    logger.info(precision)
    logger.info("recall_score:")
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    logger.info(recall)

    return f1, precision, recall

def load_data(dataset_path, split="full"):
    """
    Load the dataset from the specified path.
    Clean the text data by removing punctuation and converting to lowercase.
    Encode the labels using LabelEncoder.
    Args:
        dataset_path (str): Path to the dataset directory.
        split (str): The split of the dataset to load. Default is "full".
    Returns:
        X_train (pd.Series): Cleaned training input data.
        X_test (pd.Series): Cleaned testing input data.
        y_train_encoded (np.ndarray): Encoded training labels.
        y_test_encoded (np.ndarray): Encoded testing labels.
        le (LabelEncoder): LabelEncoder object used for encoding labels.
        num_labels (int): Number of unique labels in the dataset.
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

    return y_true, y_pred

def compute_svm(le, X_train_seq, X_test_seq, y_train_encoded, y_test_encoded):
    clf = LinearSVC(random_state=42)
    clf.fit(X_train_seq, y_train_encoded)
    pred = clf.predict(X_test_seq)
    y_true = [le.classes_[y] for y in y_test_encoded]
    y_pred = [le.classes_[y] for y in pred]

    return y_true, y_pred

def compute_lr(le, X_train_seq, X_test_seq, y_train_encoded, y_test_encoded):
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_seq, y_train_encoded)
    pred = clf.predict(X_test_seq)
    y_true = [le.classes_[y] for y in y_test_encoded]
    y_pred = [le.classes_[y] for y in pred]

    return y_true, y_pred

def compute_nb(le, X_train_seq, X_test_seq, y_train_encoded, y_test_encoded):
    clf = GaussianNB()
    clf.fit(X_train_seq.toarray(), y_train_encoded)
    pred = clf.predict(X_test_seq.toarray())
    y_true = [le.classes_[y] for y in y_test_encoded]
    y_pred = [le.classes_[y] for y in pred]

    return y_true, y_pred

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "dataset_dir",
        # default=None,
        type=str,
        # required=True,
        help="The input data dir. Should contain the training files for the text baseline task.",
    )

    # Optional parameters
    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="The output directory where the model predictions results will be written.",
    )
    
    args = parser.parse_args()

    splits = ["full", "5k_split"]
    for split in splits:

        logger.info("Run {split} benchmark script")
        
        # Load the dataset
        X_train, X_test, y_train_encoded, y_test_encoded, le, num_labels = load_data(args.dataset_dir, split=split)
        logger.info("Dataset loaded")

        # Apply Bag of Words
        X_train_seq, X_test_seq = apply_bow(X_train, X_test)
        logger.info("Bag of Words applied")
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        # Compute KNN
        y_true, y_pred = compute_knn(le, X_train_seq, X_test_seq, y_train_encoded, y_test_encoded)
        f1, precision, recall = compute_metrics(y_true, y_pred)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        logger.info("KNN computed")

        # Compute SVM
        y_true, y_pred = compute_svm(le, X_train_seq, X_test_seq, y_train_encoded, y_test_encoded)
        f1, precision, recall = compute_metrics(y_true, y_pred)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        logger.info("SVM computed")

        # Compute Logistic Regression
        y_true, y_pred = compute_lr(le, X_train_seq, X_test_seq, y_train_encoded, y_test_encoded)
        f1, precision, recall = compute_metrics(y_true, y_pred)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        logger.info("Logistic Regression computed")

        # Compute Naive Bayes
        y_true, y_pred = compute_nb(le, X_train_seq, X_test_seq, y_train_encoded, y_test_encoded)
        f1, precision, recall = compute_metrics(y_true, y_pred)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        logger.info("Naive Bayes computed")

        # Export results to CSV
        os.makedirs(args.output_dir, exist_ok=True)
        export_results_to_csv(precision_scores, recall_scores, f1_scores, args.output_dir)
        logger.info("Results exported to CSV")


if __name__ == "__main__":
    main()
    logger.info("Training script finished")
