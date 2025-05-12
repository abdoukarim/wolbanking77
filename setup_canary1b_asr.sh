#!/bin/bash

# Install Python dependencies
# pip install -U torch torchaudio --no-cache-dir
apt-get update
apt-get install -y sox libsndfile1 ffmpeg
python -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
pip install -r tasks/asr/finetune/requirements_canary1b_flash.txt
