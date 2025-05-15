import sys
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
from huggingface_hub import HfFolder

sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import set_seed, load_data, tokenize_data

set_seed(42)
logger = setup_logger("afriteva_v2 Training script")

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using {device} device")


# Metric helper method
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics: F1 score, precision, and recall.
    Args: 
    eval_pred (tuple): Tuple containing the predictions and labels.
    Returns:
        dict: Dictionary containing F1 score, precision, and recall.
    """
    metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")


# Tokenize helper function
def tokenize(batch, model_id="castorini/afriteva_v2_base"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")


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
    
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Set number of training epochs.  ",
    )

    args = parser.parse_args()

    # Model id to load the tokenizer
    model_id = "castorini/afriteva_v2_base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info("=========================== FINETUE afriteva_v2 ===========================")

    logger.info("Run {split} benchmark script".format(split=args.split))
    
    # Load the dataset
    raw_dataset = load_data(args.dataset_dir, split=args.split)
    logger.info("Dataset loaded")
    # Tokenize the dataset
    tokenized_dataset, _, num_labels, label2id, id2label = tokenize_data(raw_dataset, tokenize)

    # Download the model from huggingface.co/models
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
    
    # Id for remote repository
    repository_id = "checkpoint/afriteva_v2_base/afriteva_v2_base-banking77-wolof-{split}".format(split=args.split)

    # Define training args
    training_args = TrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=8, # 8 16
        per_device_eval_batch_size=8, # 8
        learning_rate=2e-05,
        # num_train_epochs=20,
        num_train_epochs=args.epochs, # 1,
        warmup_ratio=0.1,
        weight_decay=0.01,
            # PyTorch 2.0 specifics 
        # bf16=True, # bfloat16 training 
        # torch_compile=True, # optimizations
        optim="adamw_torch_fused", # improved optimizer 
        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=200,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
        remove_unused_columns=False

    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    logger.info("Number of epochs: {epochs}".format(epochs=args.epochs))

    # Start training
    trainer.train()

    # Save processor and create model card
    tokenizer.save_pretrained(repository_id)
    # save model
    trainer.save_model(repository_id)


if __name__ == "__main__":
    main()
    logger.info("Training afriteva_v2 finished")
