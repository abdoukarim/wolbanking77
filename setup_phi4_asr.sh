#!/bin/bash

# Install Python dependencies
wget https://github.com/Syllo/nvtop/releases/download/3.0.2/nvtop-x86_64.AppImage
chmod +x nvtop-x86_64.AppImage
apt update
apt install -y htop nano ffmpeg
MAX_JOBS=4 pip install --no-build-isolation flash-attn==2.7.3
pip install -r tasks/asr/finetune/requirements_phi4.txt
