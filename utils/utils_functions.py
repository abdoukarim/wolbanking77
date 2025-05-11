# encoding: utf-8
import os
import pandas as pd
import random
import numpy as np
import torch

from datasets import Dataset
from datasets import concatenate_datasets
from datasets import DatasetDict
from datasets import ClassLabel
from utils import logger
from transformers import BatchFeature

def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def load_data(dataset_path, split="full"):
    """
    Load the dataset from the specified path and split it into train and test sets.
    Args:
        dataset_path (str): Path to the dataset directory.
        split (str): Split of the dataset to load. Can be "full", "train", or "test".
    Returns:
        raw_dataset (DatasetDict): A dictionary containing the train and test datasets.
    """
    df_train = pd.read_csv(os.path.join(dataset_path, "{split}/train/train.csv".format(split=split)))
    df_test = pd.read_csv(os.path.join(dataset_path, "{split}/test/test.csv".format(split=split)))
    df_train = df_train[['input_wo', 'label']]
    df_test = df_test[['input_wo', 'label']]
    #removal of punctuation marks
    df_train['input_wo_clean'] = df_train['input_wo'].replace(regex=True, to_replace = '[^\\w\\s]', value = ' ')
    df_test['input_wo_clean'] = df_test['input_wo'].replace(regex=True, to_replace = '[^\\w\\s]', value = ' ')
    # Apply lowercase
    df_train['input_wo_clean'] = df_train['input_wo_clean'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df_test['input_wo_clean'] = df_test['input_wo_clean'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    train_ds = Dataset.from_pandas(df_train[['input_wo_clean', 'label']], split="train")
    test_ds = Dataset.from_pandas(df_test[['input_wo_clean', 'label']], split="test")

    raw_dataset = DatasetDict({"train": train_ds, "test": test_ds})
    raw_dataset =  raw_dataset.rename_column("input_wo_clean", "text") # to match Trainer
    
    return raw_dataset


def tokenize_data(raw_dataset, tokenize):
    # Load Tokenizer
    tokenized_dataset = raw_dataset.map(tokenize, batched=True,remove_columns=["text"])
    labels = []
    for label in tokenized_dataset["train"]:
        labels.append(label['label'])
    labels = list(set(labels))
    num_labels = len(labels)
    new_features = tokenized_dataset['train'].features.copy()
    new_features['label'] = ClassLabel(names=list(set(labels)))
    tokenized_dataset['train'] = tokenized_dataset['train'].cast(new_features)
    new_features = tokenized_dataset['test'].features.copy()
    new_features['label'] = ClassLabel(names=list(set(labels)))
    tokenized_dataset['test'] = tokenized_dataset['test'].cast(new_features)

    # Prepare model labels - useful for inference
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    return tokenized_dataset, labels, num_labels, label2id, id2label


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


INSTRUCTION = "Transcribe the Wolof audio clip."
ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100
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


