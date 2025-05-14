#!/bin/bash

# Install Python dependencies
# pip install -U torch torchaudio --no-cache-dir
apt-get update
apt-get install -y sox libsndfile1 ffmpeg
# python -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
git clone https://github.com/NVIDIA/NeMo --branch v2.3.0
cd NeMo/
# Install the required dependencies for ASR
pip install '.[asr]'
cd ..
pip install -r tasks/asr/finetune/requirements_canary1b_flash.txt
