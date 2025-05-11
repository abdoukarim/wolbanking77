import torch
import tqdm
from transformers import pipeline
import os, sys
import argparse
import pandas as pd
from sklearn import metrics
from datasets import ClassLabel

sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import set_seed, load_data, tokenize_data

set_seed(42)
logger = setup_logger("afriteva_v2_base Evaluation script")

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
        'Model': ["afriteva_v2_base"],
        'split': [split],
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    results_df = pd.DataFrame(data)
    results_df.to_csv(
        os.path.join(output_dir, "benchmark_afriteva_v2_base_results_{split}.csv".format(split=split)), index=False)


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


def create_prompt(text: str, class_names):
    prompt = """
Classify the text for one of the categories:

<text>
{text}
</text>

Choose from one of the category:
{classes}
Only choose one category, the most appropriate one. Reply only with the category.
""".strip()
    return prompt.format(text=text, classes=", ".join(class_names))


def create_dataset(df, id2label, labels):
    rows = []
    for _, row in tqdm(df.iterrows()):
        rows.append(
            {
                "input": create_prompt(row.text, labels),
                "output": id2label[str(row.label)].lower(),
            }
        )
    return rows


def predict(generator, test_rows, labels):
    predictions = []
    true_values = []
    for row in tqdm(test_rows):
        messages = [{"role": "user", "content": create_prompt(row["input"], labels)}]
        outputs = generator(
            messages, max_new_tokens=32, pad_token_id=generator.tokenizer.eos_token_id
        )
        predictions.append(outputs[0]["generated_text"][-1]["content"].lower())
        true_values.append(row["output"])
    return predictions, true_values


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

    logger.info("=========================== EVALUATE afriteva_v2_base ===========================")

    logger.info("Run {split} benchmark script".format(split=args.split))
    
    # Load the dataset
    raw_dataset = load_data(args.dataset_dir, split=args.split)
    logger.info("Dataset loaded")
    labels = []
    for label in raw_dataset["train"]:
        labels.append(label['label'])
    labels = list(set(labels))
    # Prepare model labels - useful for inference
    # num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    new_features = raw_dataset['train'].features.copy()
    new_features['label'] = ClassLabel(names=list(set(labels)))
    raw_dataset['train'] = raw_dataset['train'].cast(new_features)
    new_features = raw_dataset['test'].features.copy()
    new_features['label'] = ClassLabel(names=list(set(labels)))
    raw_dataset['test'] = raw_dataset['test'].cast(new_features)    
    # Tokenize the dataset
    # tokenized_dataset, _, num_labels, label2id, id2label = tokenize_data(raw_dataset, tokenize)
    # train_df = pd.DataFrame(raw_dataset['train'])
    test_df = pd.DataFrame(raw_dataset['test'])

    # train_rows = create_dataset(train_df, id2label, labels)
    test_rows = create_dataset(test_df, id2label, labels)

    model_id = os.path.join(os.getcwd(), "Llama-3.2-1B-Instruct-torchtune-checkpoints{split}".format(split=args.split))
    # model_id = "/workspace/NLP_TASKS/Llama-3.2-1B-Instruct-torchtune-checkpoints/epoch_14"
    generator = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    predictions, true_values = predict(generator, test_rows, labels)
    f1, precision, recall = compute_metrics(true_values, predictions)
    # Export the results to a CSV file
    export_results_to_csv(precision, recall, f1, args.output_dir, args.split)
    

if __name__ == "__main__":
    main()
    logger.info("Llama3.2 evaluation finished")

