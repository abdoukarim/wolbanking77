# Core Imports
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer
)

from accelerate import Accelerator
from pathlib import Path
import torch
import os, sys
import argparse

from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from evaluate import load
wer_metric = load("wer")

sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import WolBanking77Dataset, esb_collate_fn, set_seed

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
    # ds = load_from_disk(args.dataset_dir)
    ds = load_dataset("parquet", 
                      data_files={'train': os.path.join(args.dataset_dir, 'train.parquet'), 
                                  'test': os.path.join(args.dataset_dir, 'test.parquet')})

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
    # val_dataset = WolBanking77Dataset(processor, val_ds, training=False)

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
