import argparse
from pathlib import Path
import os, sys
from datasets import load_dataset
import soundfile as sf
import pandas as pd
import numpy as np
import json
import numpy as np
from omegaconf import OmegaConf
import subprocess
from nemo.collections.asr.models import EncDecMultiTaskModel


sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import set_seed

set_seed(42)
logger = setup_logger("Canary 1b ASR Training script")


def build_manifest(data, manifest_path):
    """
    Build the manifest file for the dataset.
    Args:
        data: the dataset to build the manifest from
        manifest_path: the path to save the manifest file in JSON format
    """
    tot_duration = 0
    for line in data.iterrows():
        with open(manifest_path, 'a') as fout:
            audio_path = "./dataset/audio/wavs/"+line[1][1]['path']
            duration = line[1][3]
            transcript = line[1][0]
            tot_duration += duration
            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": audio_path,
                "duration": duration/1000, # duration to second
                "text": transcript,
                "target_lang": "wo",
                "source_lang": "wo",
                "lang": "wo",
                "pnc": "False"
            }
            json.dump(metadata, fout)
            fout.write('\n')
    print(f'\n{np.round((tot_duration/1000)/3600)} hour audio data ready for training')


def preprocess_audio_dataset(example):
    """
    Preprocess the audio dataset by loading it and saving the audio files to dataset/audio/wavs directory.
    """
    Path("./dataset/audio/wavs").mkdir(parents=True, exist_ok=True)
    # Load the dataset
    sf.write("./dataset/audio/wavs/"+example['audio']['path'],
         example['audio']["array"], samplerate=example['audio']["sampling_rate"])


BRANCH='main'
def wget_from_nemo(nemo_script_path, local_dir="scripts"):
    os.makedirs(local_dir, exist_ok=True)
    script_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/refs/heads/{BRANCH}/{nemo_script_path}"
    script_path = os.path.basename(nemo_script_path)
    if not os.path.exists(f"{local_dir}/{script_path}"):
        subprocess.Popen("wget -P {local_dir}/ {script_url}".format(local_dir=local_dir,script_url=script_url), shell=True)
        

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
    
    args = parser.parse_args()

    ds = load_dataset("parquet", 
                      data_files={'train': os.path.join(args.dataset_dir, 'train.parquet'), 
                                  'test': os.path.join(args.dataset_dir, 'test.parquet')})
    ds['train'].map(preprocess_audio_dataset)
    ds['test'].map(preprocess_audio_dataset)
    manifest_path = './train_manifest.json'
    build_manifest(data=pd.DataFrame(ds['train']), manifest_path=manifest_path)
    wget_from_nemo('scripts/tokenizers/process_asr_text_tokenizer.py')
    LANG='wo'
    DATA='WICD'
    VOCAB_SIZE=1024
    OUT_DIR = f"tokenizers/{LANG}_{DATA}_{VOCAB_SIZE}"
    # create train data
    manifest_path = './train_manifest.json'
    train_text_path = './train_text.lst'
    with open(manifest_path, "r") as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    with open(train_text_path, "w") as f:
        for line in data:
            f.write(f"{line['text']}\n")
    
    subprocess.Popen("""
        python scripts/process_asr_text_tokenizer.py \
        --data_file={train_text_path} \
        --vocab_size={VOCAB_SIZE} \
        --data_root={OUT_DIR} \
        --tokenizer="spe" \
        --spe_type=bpe \
        --spe_character_coverage=1.0 \
        --no_lower_case \
        --log
    """.format(
            train_text_path=train_text_path,
            VOCAB_SIZE=VOCAB_SIZE,
            OUT_DIR=OUT_DIR
        ),
        shell=True
    )

    wget_from_nemo('examples/asr/speech_multitask/speech_to_text_aed.py')
    wget_from_nemo('examples/asr/conf/speech_multitask/fast-conformer_aed.yaml', 'config')

    if 'canary_model' not in locals():
        canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-flash') 
    
    base_model_cfg = OmegaConf.load("config/fast-conformer_aed.yaml")
    base_model_cfg['name'] = 'canary-1b-flash-finetune'
    base_model_cfg.pop("init_from_nemo_model", None)
    base_model_cfg['init_from_pretrained_model'] = "nvidia/canary-1b-flash"
    # canary_model.save_tokenizers('./canary_flash_tokenizers/')

    for lang in os.listdir('canary_flash_tokenizers'):
        base_model_cfg['model']['tokenizer']['langs'][lang] = {}
        base_model_cfg['model']['tokenizer']['langs'][lang]['dir'] = os.path.join('canary_flash_tokenizers', lang)
        base_model_cfg['model']['tokenizer']['langs'][lang]['type'] = 'bpe'
    
    base_model_cfg['spl_tokens']['model_dir'] = os.path.join('canary_flash_tokenizers', "spl_tokens")
    base_model_cfg['model']['prompt_format'] = canary_model._cfg['prompt_format']
    base_model_cfg['model']['prompt_defaults'] = canary_model._cfg['prompt_defaults']
    base_model_cfg['model']['model_defaults'] = canary_model._cfg['model_defaults']
    base_model_cfg['model']['preprocessor'] = canary_model._cfg['preprocessor']
    base_model_cfg['model']['encoder'] = canary_model._cfg['encoder']
    base_model_cfg['model']['transf_decoder'] = canary_model._cfg['transf_decoder']
    base_model_cfg['model']['transf_encoder'] = canary_model._cfg['transf_encoder']

    # create test data
    manifest_path = './test_manifest.json'
    build_manifest(data=pd.DataFrame(ds['test']), manifest_path=manifest_path)

    MANIFEST = './train_manifest.json'
    MANIFEST_TEST = './test_manifest.json'

    # run the training script
    subprocess.Popen("""
        HYDRA_FULL_ERROR=1 python scripts/speech_to_text_aed.py \
            --config-path="../config" \
            --config-name="canary-1b-flash-finetune.yaml" \
            name="canary-1b-flash-finetune" \
            model.train_ds.manifest_filepath={MANIFEST} \
            model.validation_ds.manifest_filepath={MANIFEST} \
            model.test_ds.manifest_filepath={MANIFEST_TEST} \
            exp_manager.exp_dir="canary_results" \
            exp_manager.resume_ignore_no_checkpoint=true \
            trainer.max_steps=10 \
            trainer.log_every_n_steps=1
    """.format(
            MANIFEST=MANIFEST,
            MANIFEST_TEST=MANIFEST_TEST
        ),
        shell=True
    )


if __name__ == "__main__":
    main()
    logger.info("Training Canary1b finished")
