import argparse
import os, sys

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor
from datasets import load_dataset
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import set_seed

set_seed(42)
logger = setup_logger("distil-large-v3.5 ASR Eval script")
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using {device} device")

feature_extractor = WhisperFeatureExtractor.from_pretrained("distil-whisper/distil-large-v3.5")
tokenizer = WhisperTokenizer.from_pretrained("distil-whisper/distil-large-v3.5", task="transcribe")


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


def wer_evaluate(predictions, references):
    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=predictions, references=references)
    return wer


def predictions(pipe, example):
    sample = example["audio"]
    result = pipe(sample)
    return {'text': result['text']}


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

    ds = load_dataset("parquet", 
                      data_files={'train': os.path.join(args.dataset_dir, 'train.parquet'), 
                                  'test': os.path.join(args.dataset_dir, 'test.parquet')})
    dataset_train = ds['train']
    dataset_test = ds['test']
    dataset_test = dataset_test.map(prepare_dataset)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "./distil_whisper_checkpoints"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    # processor = AutoProcessor.from_pretrained(model_id)
    processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v3.5", task="transcribe")
    # tokenizer = WhisperTokenizer.from_pretrained("distil-whisper/distil-large-v3.5", task="transcribe")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer, # processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )

    preds = dataset_test.map(lambda x: predictions(pipe, x))
    references = dataset_test.map(lambda x: {'text': x["text"]})
    wer = wer_evaluate(preds["text"], references["text"])
    sample = dataset_test[0]["audio"]

    logger.info(f"WER: {wer}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

