import torch
import argparse
import sys, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import ClassLabel
from sklearn import metrics
import pandas as pd

sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import set_seed, load_data

set_seed(42)
logger = setup_logger("Bert Base Evaluation script")

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using {device} device")


def export_results_to_csv(precision, recall, f1, output_dir, split):
    """
    Export the results to a CSV file.
    Args:
        precision (list): List of precision scores.
        recall (list): List of recall scores.
        f1 (list): List of F1 scores.
        output_dir (str): Directory to save the CSV file.
        split (str): The split of the dataset.
    """
    # Save the DataFrame to a CSV file
    
    data = {
        'Model': ["BERT"], # ['LASER+MLP'],
        'split': [split],
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    results_df = pd.DataFrame(data)
    results_df.to_csv(
        os.path.join(output_dir, "benchmark_BERT_results_{split}.csv".format(split=split)), index=False)


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

# Tokenize helper function
def tokenize(batch, model_id="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")


def predict(batch, model):
    input_ids = torch.tensor(batch['input_ids']).to(device).unsqueeze(0)
    token_type_ids = torch.tensor(batch['token_type_ids']).to(device).unsqueeze(0)
    attention_mask = torch.tensor(batch['attention_mask']).to(device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
        predicted = torch.argmax(logits, dim=-1)[0]
    return predicted


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

    parser.add_argument(
        "split",
        type=str,
        help="The split of the dataset to load. Can be 'full' or '5k_split'.",
    )

    # Optional parameters
    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="The output directory where the model predictions results will be written.",
    )
    
    args = parser.parse_args()
    # check if the output directory exists, if not create it
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=========================== EVALUATE BERT BASE ===========================")

    logger.info("Run {split} benchmark script".format(split=args.split))
    
    # Load the dataset
    raw_dataset = load_data(args.dataset_dir, split=args.split)
    logger.info("Dataset loaded")
    # Load Tokenizer
    tokenized_dataset = raw_dataset.map(tokenize, batched=True,remove_columns=["text"])
    labels = []
    for label in tokenized_dataset["train"]:
        labels.append(label['label'])
    labels = list(set(labels))

    new_features = tokenized_dataset['train'].features.copy()
    new_features['label'] = ClassLabel(names=list(set(labels)))
    tokenized_dataset['train'] = tokenized_dataset['train'].cast(new_features)
    new_features = tokenized_dataset['test'].features.copy()
    new_features['label'] = ClassLabel(names=list(set(labels)))
    tokenized_dataset['test'] = tokenized_dataset['test'].cast(new_features)

    # Prepare model labels - useful for inference
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # repository_id = os.path.join(os.getcwd(), "checkpoint/BERT/BERT-base-banking77-wolof-{split}".format(split=args.split)+"/checkpoint-125")
    repository_id = os.path.join(os.getcwd(), "checkpoint/BERT/BERT-base-banking77-wolof-{split}".format(split=args.split))
    # model_id = repository_id
    # tokenizer = AutoTokenizer.from_pretrained(repository_id)
    model = AutoModelForSequenceClassification.from_pretrained(repository_id, use_safetensors=True).to("cuda")
    model.to("cuda")
    predictions = [predict(batch, model) for batch in tokenized_dataset["test"]]
    y_pred = []
    for y in predictions:
        y_pred.append(labels[y])
    
    y_test = []
    for y in tokenized_dataset["test"]['label']:
        y_test.append(labels[y])
    
    f1, precision, recall = compute_metrics(y_test, y_pred)
    # Export the results to a CSV file
    export_results_to_csv(precision, recall, f1, args.output_dir, args.split)
    
if __name__ == "__main__":
    main()
    logger.info("Bert evaluation finished")
