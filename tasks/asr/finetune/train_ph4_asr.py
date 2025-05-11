# Core Imports
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BatchFeature,
    StoppingCriteria,
)

from accelerate import Accelerator
from pathlib import Path
import torch
import sys
import argparse

from datasets import load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from evaluate import load
wer_metric = load("wer")

sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import set_seed, load_data, tokenize_data

set_seed(42)
logger = setup_logger("Phi4 ASR Training script")

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using {device} device")


# Fixed ASR instruction and other constants
INSTRUCTION = "Transcribe the Wolof audio clip."
ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100

def normalize_text(text):
    """
    Placeholder for text normalization. You can use whisper text normalizer/jiwer or similar tools.
    """
    return text


class WolBanking77Dataset(Dataset):
    def __init__(self, processor, dataset, training=True):
        """
        processor: the AutoProcessor instance
        dataset: a Hugging Face Dataset (already split into train/validation)
        training: whether this dataset is for training (affects concatenation of target tokens)
        """
        self.data = dataset
        self.training = training
        self.processor = processor
        self.instruction = INSTRUCTION

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # The dataset contains an "audio" dict and a "text" field for transcription.
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + self.instruction,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            audios=[(data["audio"]["array"], data["audio"]["sampling_rate"])],
            return_tensors='pt'
        )
        
        answer = f"{data['text']}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        if self.training:
            # Concatenate prompt and answer, but mask all tokens except the answer.
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1]:] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_audio_embeds': inputs.input_audio_embeds,
            'audio_embed_sizes': inputs.audio_embed_sizes,
        }

def pad_sequence(sequences, padding_side='right', padding_value=0):
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    ndim = tensors[0].dim()
    assert all(t.dim() == ndim for t in tensors[1:]), 'All tensors must have the same number of dimensions'
    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)
    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[slices] = t
        index += t.shape[dim]
    return output


def esb_collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        audio_attention_mask_list.append(
            inputs['input_audio_embeds'].new_full((inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool)
        )
    try:
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=0)
        audio_attention_mask = (
            pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
            if len(audio_attention_mask_list) > 1 else None
        )
    except Exception as e:
        logger.error(e)
        logger.info(input_ids_list)
        logger.info(labels_list)
        raise
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)
    return BatchFeature({
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'input_audio_embeds': input_audio_embeds,
        'audio_embed_sizes': audio_embed_sizes,
        'audio_attention_mask': audio_attention_mask,
        'input_mode': 2,  # speech mode
    })


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

    BATCH_SIZE_PER_GPU = 16 # 8
    # EVAL_BATCH_SIZE_PER_GPU = 24

    # Load and split the dataset.
    train_ds = ds['train']
    val_ds = ds['test']

    # num_processes = 8
    logger.info(f"Training dataset size: {len(train_ds)}")
    logger.info(f"Val dataset size: {len(val_ds)}")

    # Configuration variables
    MODEL_NAME = 'microsoft/Phi-4-multimodal-instruct'
    OUTPUT_DIR = './Phi4_mm_asr_wolbanking77_unf'
    NEW_MODEL_ID = "./Phi-4-mm-inst-asr-wolbanking77-unf-model"
    USE_FLASH_ATTENTION = True
    # BATCH_SIZE_PER_GPU = 8  See dataset loader cell for these parameters.
    # EVAL_BATCH_SIZE_PER_GPU = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.005

    # Initialize Accelerator
    accelerator = Accelerator()
    num_gpus = accelerator.num_processes
    logger.info(f"Training on {num_gpus} GPUs")

    def print_model_structure(model, max_depth=3):
        """Prints model structure up to specified depth"""
        logger.info("\n=== MODEL ARCHITECTURE ===")
        for name, module in model.named_modules():
            depth = name.count('.')
            if depth < max_depth:
                logger.info(f"{'  ' * depth}{name} ({type(module).__name__})")

    def create_model(model_name, use_flash_attention):
        """Initialize model with audio enabled"""
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            audio_enabled=True
        )
        if use_flash_attention:
            config._attn_implementation = "flash_attention_2"
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(accelerator.device)

    # --------------------------------------------------
    # Model Initialization and Unfreezing
    # --------------------------------------------------
    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = create_model(MODEL_NAME, USE_FLASH_ATTENTION)

        def unfreeze_speech_components(model):
            """Directly target verified components from your debug logs"""
            # 1. Audio Embed Module (confirmed exists)
            audio_embed = model.model.embed_tokens_extend.audio_embed

            # 2. Entire Audio Encoder (simplified)
            audio_encoder = audio_embed.encoder  # Direct access

            # 3. Audio Projection (from debug logs)
            audio_projection = audio_embed.audio_projection

            # Unfreeze ONLY these 3 components
            for component in [audio_embed, audio_encoder, audio_projection]:
                for param in component.parameters():
                    param.requires_grad = True
            return model

        model = unfreeze_speech_components(model)

        # Verify unfrozen parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info("Unfrozen components:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"- {name}")

        # After unfreezing
        encoder_params = list(model.model.embed_tokens_extend.audio_embed.encoder.parameters())
        proj_params = list(model.model.embed_tokens_extend.audio_embed.audio_projection.parameters())

        assert any(p.requires_grad for p in encoder_params), "Encoder params frozen!"
        assert any(p.requires_grad for p in proj_params), "Projection params frozen!"
        logger.info("Components properly unfrozen ✅")

    # Create dataset objects.
    train_dataset = WolBanking77Dataset(processor, train_ds, training=True)
    val_dataset = WolBanking77Dataset(processor, val_ds, training=False)

    # --------------------------------------------------
    # Optimizer Configuration with Correct Gradient Handling
    # --------------------------------------------------
    gradient_accumulation_steps = max(1, BATCH_SIZE_PER_GPU // (BATCH_SIZE_PER_GPU // num_gpus))
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")


    # Set mixed precision flags.
    fp16 = not USE_FLASH_ATTENTION
    bf16 = USE_FLASH_ATTENTION

    # --------------------------------------------------
    # Training Preparation with DDP Fixes
    # --------------------------------------------------
    training_args = TrainingArguments(
        gradient_accumulation_steps=gradient_accumulation_steps,
        ddp_find_unused_parameters=True,  # for unused SigLIP layers
        overwrite_output_dir=True,
        save_steps=10000,
        # max_steps=1000,
        max_steps=10,
        per_device_train_batch_size=BATCH_SIZE_PER_GPU,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=1e-7,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        logging_steps=100,
        output_dir=OUTPUT_DIR,
        save_strategy='epoch',
        save_total_limit=2,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        push_to_hub=False,
        # hub_private_repo=True,
        report_to="tensorboard",
        hub_model_id=NEW_MODEL_ID
    )
    #--------------------------------------------------

    logger.info("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=esb_collate_fn,
    )

    trainer.train()

    # Save full model with processor and configs
    trainer.save_model(OUTPUT_DIR)
    # processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
    logger.info("Training Phi4 finished")
