import os, sys
import argparse
from datasets import load_from_disk #, load_dataset
import pandas as pd
import random
import numpy as np
import torch

from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate


sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import set_seed

set_seed(42)
logger = setup_logger("Whisper ASR Training script")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def prepare_dataset(batch):
    # load audio data
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def compute_metrics(pred):
    metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


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
    
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_dir)
    dataset_train = ds['train']
    dataset_test = ds['test']
    feature_extractor = WhisperFeatureExtractor.from_pretrained("distil-whisper/distil-large-v3.5")
    tokenizer = WhisperTokenizer.from_pretrained("distil-whisper/distil-large-v3.5", task="transcribe")
    processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v3.5", task="transcribe")
    dataset_train = dataset_train.map(prepare_dataset)
    dataset_test = dataset_test.map(prepare_dataset)

    model = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3.5")
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"

    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./distil_whisper_checkpoints/", # replace with correct path
        per_device_train_batch_size=8, # 16
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        # max_steps=2000,
        max_steps=10,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model("./distil_whisper_checkpoints/")


if __name__ == "__main__":
    main()
    logger.info("Training Whisper finished")
