import pandas as pd
# from matplotlib import pylab as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse

from laser_encoders import LaserEncoderPipeline

sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import set_seed

set_seed(42)
logger = setup_logger("Baseline benchmark script")

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using {device} device")


class MLP(nn.Module):
    def __init__(self, embed_size, num_labels):
        """
        Initialize the MLP model.
        Args:
            embed_size (int): Size of the input embeddings.
            num_labels (int): Number of output labels.
        """
        super(MLP, self).__init__()
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embed_size, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, num_labels),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class CNNModel(nn.Module):
  def __init__(self, embed_size, num_labels):
    """
    Initialize the CNN model.
    Args:
        embed_size (int): Size of the input embeddings.
        num_labels (int): Number of output labels.
    """
    super().__init__()
    self.conv = nn.Conv1d(embed_size, 1, kernel_size=1, stride=1, padding=1)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(3, num_labels)

  def forward(self, x):
    x = x.transpose(0, 1)
    conv_layer = self.relu(self.conv(x))
    return self.fc(conv_layer)  


def export_results_to_csv(precision, recall, f1, output_dir, split, is_dl=False):
    """
    Export the results to a CSV file.
    Args:
        precision (list): List of precision scores.
        recall (list): List of recall scores.
        f1 (list): List of F1 scores.
        output_dir (str): Directory to save the CSV file.
        split (str): The split of the dataset.
        is_dl (bool): Flag to indicate if the results are from DL models.
    """
    # Save the DataFrame to a CSV file
    if is_dl:
        data = {
            'Model': ["LASER+MLP", "LASER+CNN"], # ['LASER+MLP'],
            'split': [split, split],
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
        results_df = pd.DataFrame(data)
        results_df.to_csv(
            os.path.join(output_dir, "benchmark_DL_baseline_results_{split}.csv".format(split=split)), index=False)
    else:
        data = {
            'Model': ['BoW+KNN', 'BoW+SVM', 'BoW+LR', 'BoW+NB'],
            'split': [split, split, split, split],
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
        results_df = pd.DataFrame(data)
        results_df.to_csv(os.path.join(output_dir, "benchmark_ml_baseline_results_{split}.csv".format(split=split)), index=False)


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics: F1 score, precision, and recall.
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
    Returns:
        f1 (float): F1 score.
        precision (float): Precision score.
        recall (float): Recall score.
    """
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


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    avg_loss = 0.
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            avg_loss += loss / len(dataloader)
    return avg_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


def compute_laser_mlp(embed_size, train_loader, valid_loader, y_test_encoded, num_labels):
    mlp_model = MLP(embed_size, num_labels).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=1e-3)
    epochs = 200
    # epochs = 1
    train_loss = []
    test_avg_loss = []
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train_loss.append(train(train_loader, mlp_model, loss_fn, optimizer))
        test_avg_loss.append(test(valid_loader, mlp_model, loss_fn))
    logger.info("Mlp Training Done!")

    return mlp_model, valid_loader, loss_fn, y_test_encoded


def compute_laser_cnn(embed_size, train_loader, valid_loader, y_test_encoded, num_labels):
    cnn_model = CNNModel(embed_size, num_labels).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=1e-3)
    epochs = 200
    # epochs = 1

    train_loss = []
    test_avg_loss = []
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train_loss.append(train(train_loader, cnn_model, loss_fn, optimizer))
        test_avg_loss.append(test(valid_loader, cnn_model, loss_fn))
    logger.info("Done!")

    return cnn_model, valid_loader, loss_fn, y_test_encoded


def eval_laser_mlp(mlp_model, valid_loader, loss_fn, le, y_test_encoded):
    """
    Evaluate the MLP model on the validation set.
    Args:
        mlp_model (MLP): The trained MLP model.
        valid_loader (DataLoader): DataLoader for the validation set.
        loss_fn (nn.Module): Loss function used for evaluation.
    """
    # MODEL EVALUATION
    mlp_model.eval()
    avg_val_loss = 0.
    val_preds = []
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = mlp_model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        # keep/store predictions
        val_preds.append(F.softmax(y_pred).cpu().numpy().argmax(1))
    y_true = [le.classes_[y] for y in y_test_encoded]
    y_pred = [le.classes_[y] for y in val_preds]

    return y_true, y_pred


def eval_cnn_mlp(cnn_model, valid_loader, loss_fn, le, y_test_encoded):
    """
    Evaluate the CNN model on the validation set.
    Args:
        mlp_model (MLP): The trained MLP model.
        valid_loader (DataLoader): DataLoader for the validation set.
        loss_fn (nn.Module): Loss function used for evaluation.
    """
    # MODEL EVALUATION
    cnn_model.eval()
    avg_val_loss = 0.
    val_preds = []
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = cnn_model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        # keep/store predictions
        val_preds.append(F.softmax(y_pred).cpu().numpy().argmax(1))
    y_true = [le.classes_[x] for x in y_test_encoded]
    y_pred = [le.classes_[x] for x in val_preds]

    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "dataset_dir",
        # default=None,
        type=str,
        # required=True,
        help="The input data dir.  ",
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
    
    logger.info("=========================== Compute ML BENCHMARKS ===========================")
    for split in splits:

        logger.info("Run {split} benchmark script".format(split=split))
        
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
        export_results_to_csv(precision_scores, recall_scores, f1_scores, args.output_dir, split)
        logger.info("Results exported to CSV")
    
    logger.info("=========================== Compute LASER BENCHMARKS ===========================")
    os.makedirs("checkpoint/laser_checkpoint", exist_ok=True)
    encoder = LaserEncoderPipeline(lang="wol_Latn", model_dir="checkpoint/laser_checkpoint")
    for split in splits:

        logger.info("Run {split} benchmark script".format(split=split))
        
        # Load the dataset
        X_train, X_test, y_train_encoded, y_test_encoded, le, num_labels = load_data(args.dataset_dir, split=split)
        logger.info("Dataset loaded")
        
        X_train_seq = encoder.encode_sentences(list(X_train))
        X_test_seq = encoder.encode_sentences(list(X_test))

        embed_size = X_train_seq.shape[1]
        # Load train and test in CUDA Memory
        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
        X_valid = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        y_valid = torch.tensor(y_test_encoded, dtype=torch.long).to(device)

        # Create Torch datasets
        train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        valid_data = torch.utils.data.TensorDataset(X_valid, y_valid)

        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False)

        f1_scores = []
        precision_scores = []
        recall_scores = []
        # Compute MLP+CNN
        mlp_model, valid_loader, loss_fn, y_test_encoded = compute_laser_mlp(embed_size, train_loader, valid_loader, y_test_encoded, num_labels)
        logger.info("MLP computed")
        y_true, y_pred = eval_laser_mlp(mlp_model, valid_loader, loss_fn, le, y_test_encoded)
        f1, precision, recall = compute_metrics(y_true, y_pred)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        logger.info("MLP evaluation done")

        # Compute LASER+CNN
        cnn_model, valid_loader, loss_fn, y_test_encoded = compute_laser_cnn(embed_size, train_loader, valid_loader, y_test_encoded, num_labels)
        logger.info("CNN computed")
        y_true, y_pred = eval_cnn_mlp(cnn_model, valid_loader, loss_fn, le, y_test_encoded)
        f1, precision, recall = compute_metrics(y_true, y_pred)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        logger.info("CNN evaluation done")
        
        # Export results to CSV
        export_results_to_csv(precision_scores, recall_scores, f1_scores, args.output_dir, split, is_dl=True)
        logger.info("Results exported to CSV")
        


if __name__ == "__main__":
    main()
    logger.info("Training script finished")
