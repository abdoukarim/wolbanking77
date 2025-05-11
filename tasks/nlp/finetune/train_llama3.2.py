import sys
import json
import re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

import argparse
from transformers import pipeline
import evaluate

from datasets import ClassLabel

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


def create_prompt(text: str, class_names: List[str]):
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


    logger.info("=========================== FINETUNE Llama3.2 ===========================")

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
    train_df = pd.DataFrame(raw_dataset['train'])
    test_df = pd.DataFrame(raw_dataset['test'])

    train_rows = create_dataset(train_df, id2label, labels)
    test_rows = create_dataset(test_df, id2label, labels)

    # Format dataset for finetune
    Path("train_data.json").write_text(json.dumps(train_rows))
    Path("test_data.json").write_text(json.dumps(test_rows))

    # Download the model 
    """
    !tune download "meta-llama/Llama-3.2-1B-Instruct" \
    --output-dir "./Llama-3.2-1B-Instruct" \
    --hf-token "{hf_token}" \
    --ignore-patterns "[]"
    """

    config = """
    # Model Arguments
    model:
    _component_: torchtune.models.llama3_2.lora_llama3_2_1b
    lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']
    apply_lora_to_mlp: True
    apply_lora_to_output: False
    lora_rank: 64
    lora_alpha: 128
    lora_dropout: 0.0

    # Tokenizer
    tokenizer:
    _component_: torchtune.models.llama3.llama3_tokenizer
    path: ./Llama-3.2-1B-Instruct/original/tokenizer.model
    max_seq_len: null

    checkpointer:
    _component_: torchtune.training.FullModelHFCheckpointer
    checkpoint_dir: ./Llama-3.2-1B-Instruct
    checkpoint_files: [
        model.safetensors
    ]
    recipe_checkpoint: null
    output_dir: ./Llama-3.2-1B-Instruct-torchtune-checkpoints
    model_type: LLAMA3_2
    resume_from_checkpoint: False
    save_adapter_weights_only: False

    # Dataset and Sampler
    dataset:
    _component_: torchtune.datasets.instruct_dataset
    data_files: ./train_data.json
    source: json
    split: train
    seed: 42
    shuffle: True
    # batch_size: 1
    batch_size: 4

    # Optimizer and Scheduler
    optimizer:
    _component_: torch.optim.AdamW
    fused: True
    weight_decay: 0.01
    lr: 3e-4
    lr_scheduler:
    _component_: torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup
    num_warmup_steps: 100

    loss:
    _component_: torchtune.modules.loss.LinearCrossEntropyLoss

    # Training
    epochs: 1
    max_steps_per_epoch: null
    gradient_accumulation_steps: 4
    compile: False # set it to True for better memory and performance
    # compile: True # set it to True for better memory and performance

    # Logging
    output_dir: ./logs
    metric_logger:
    _component_: torchtune.training.metric_logging.TensorBoardLogger
    log_dir: ${output_dir}
    log_every_n_steps: 1
    log_peak_memory_stats: False

    # Environment
    device: cuda
    dtype: bf16

    # Activations Memory
    enable_activation_checkpointing: False
    enable_activation_offloading: False

    # Profiler (disabled)
    profiler:
    _component_: torchtune.training.setup_torch_profiler
    enabled: False

    #Output directory of trace artifacts
    output_dir: ${output_dir}/profiling_outputs

    #`torch.profiler.ProfilerActivity` types to trace
    cpu: True
    cuda: True

    #trace options passed to `torch.profiler.profile`
    profile_memory: False
    with_stack: False
    record_shapes: True
    with_flops: False

    # `torch.profiler.schedule` options:
    # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
    wait_steps: 5
    warmup_steps: 5
    active_steps: 2
    num_cycles: 1
    """
    Path("custom_config.yaml").write_text(config)

    # create the output directory
    Path("./Llama-3.2-1B-Instruct-torchtune-checkpoints").mkdir(parents=True, exist_ok=True)
    # run following command to start training
    # !tune run lora_finetune_single_device --config "custom_config.yaml" epochs=20


if __name__ == "__main__":
    main()
    logger.info("Training Llama3.2")
