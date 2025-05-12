import argparse
import os, sys

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import evaluate

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


def wer_evaluate(predictions, references):
    metric = evaluate.load("wer")
    # sample = example["audio"]
    # result = pipe(sample)
    wer = 100 * metric.compute(predictions=predictions, references=references)
    return wer


def predictions(pipe, example):
    sample = example["audio"]
    result = pipe(sample)
    return result['text']


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

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "./distil_whisper_checkpoints"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )

    preds = dataset_test.map(lambda x: predictions(pipe, x))
    references = dataset_test.map(lambda x: x["text"])
    wer = wer_evaluate(preds, references)
    logger.info(f"WER: {wer}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

