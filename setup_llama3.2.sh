#!/bin/bash

# Install Python dependencies
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install torchao
git clone https://github.com/abdoukarim/torchtune.git
cd torchtune/
pip install -e .
cd ..
pip install -r tasks/nlp/finetune/requirements_llama3.2.txt 

