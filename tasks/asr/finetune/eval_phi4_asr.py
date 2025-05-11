import argparse
import os, sys

import torch
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
from accelerate.utils import gather_object
from evaluate import load
wer_metric = load("wer")
import json
from pathlib import Path

sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import WolBanking77Dataset, esb_collate_fn, set_seed

set_seed(42)
logger = setup_logger("Phi4 ASR Eval script")

ANSWER_SUFFIX = "<|end|><|endoftext|>"

def normalize_text(text):
    """
    Placeholder for text normalization. You can use whisper text normalizer/jiwer or similar tools.
    """
    return text


class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]
        return torch.all(self.stop_tokens_idx)


@torch.no_grad()
def evaluate(model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    model.eval()
    all_generated_texts = []
    all_labels = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=esb_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=8,  # 2 8
        prefetch_factor=32,  # 128
        pin_memory=True,
        persistent_workers=True  # Keep workers alive between batches
    )
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt")["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(f'cuda:{local_rank}')

    # with torch.cuda.amp.autocast(enabled=True):
    for inputs in tqdm(eval_dataloader, disable=(rank != 0) or disable_tqdm, desc='running eval'):
        stopping_criteria = StoppingCriteriaList([MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=inputs.input_ids.size(0))])
        inputs = inputs.to(f'cuda:{local_rank}')
        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64,
            stopping_criteria=stopping_criteria,
        )

        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(inputs.input_ids.size(0), -1)[:, 0]
        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        generated_text = [
            processor.decode(_pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx],
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]

        all_generated_texts.extend(generated_text)
        labels = [processor.decode(_label_ids[_label_ids != 0]).rstrip(ANSWER_SUFFIX) for _label_ids in inputs["labels"]]  # ⚠ See annd apply: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/33
        all_labels.extend(labels)

    all_generated_texts = gather_object(all_generated_texts)
    all_labels = gather_object(all_labels)

    if rank == 0:
        norm_all_labels = normalize_text(all_labels)
        norm_all_generated_texts = normalize_text(all_generated_texts)
        # wer = jiwer.wer(norm_all_labels, norm_all_generated_texts)
        wer = wer_metric.compute(references=norm_all_labels, predictions=norm_all_generated_texts)
        logger.info("WER:", wer)
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'all_generated_texts': all_generated_texts,
                    'all_labels': all_labels,
                    'wer': wer,
                }
                json.dump(save_dict, f)
        return wer
    return None

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
    
    OUTPUT_DIR = './Phi4_mm_asr_wolbanking77_unf'
    EVAL_BATCH_SIZE_PER_GPU = 24
    # Free up memory before re-loading the model.
    # del model, trainer
    torch.cuda.empty_cache()

    MODEL_NAME = 'microsoft/Phi-4-multimodal-instruct'
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ds = load_dataset("parquet", 
                      data_files={'train': os.path.join(args.dataset_dir, 'train.parquet'), 
                                  'test': os.path.join(args.dataset_dir, 'test.parquet')})
    val_ds = ds['test']
    val_dataset = WolBanking77Dataset(processor, val_ds, training=False)
    
    # Reload the fine-tuned model.
    model = AutoModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
        trust_remote_code=True,
        # torch_dtype='auto',
        torch_dtype=torch.bfloat16,
        _attn_implementation='flash_attention_2',
    ).cuda()
    model = torch.compile(model)
    model.eval()  # Ensure evaluation mode.

    # Evaluate the model after fine-tuning.
    logger.info("Evaluating after fine-tuning...")
    wer_after = evaluate(
        model,
        processor,
        val_dataset,
        save_path=Path(OUTPUT_DIR) / 'eval_wer.json',
        eval_batch_size=EVAL_BATCH_SIZE_PER_GPU,
    )
    logger.info(f"WER after fine-tuning: {wer_after}")


if __name__ == "__main__":
    main()

    