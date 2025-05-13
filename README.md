# Wolbanking77: Wolof Banking Speech Intent Classification Dataset

Paper link [My Paper Title](https://arxiv.org/). 

# Abstract
Intent classification models have made a lot of progress in recent years. However, previous studies primarily focus on high-resource languages datasets, which results in a gap for low-resource languages and for regions with a high rate of illiterate people where languages are more spoken than read or written. This is the case in Senegal, for example, where Wolof is spoken by around 90\% of the population, with an illiteracy rate of 42\% for the country. Wolof is actually spoken by more than 10 million people in West African region. To tackle such limitations, we release a Wolof Intent Classification Dataset (WolBanking77), for academic research in intent classification. WolBanking77 currently contains 9,791 text sentences in the banking domain and more than 4 hours of spoken sentences. Experiments on various baselines are conducted in this work, including text and voice state-of-the-art models. The results are very promising on this current dataset. This paper also provide detailed analyses of the contents of the data. We report baseline f1-score and word error rate metrics respectively on NLP and ASR models trained on WolBanking77 dataset and also comparisons between models. We plan to share and conduct dataset maintenance, updates and to release open-source code.

# Getting Started
You can download a copy of the dataset (distributed under the CC-BY-4.0 license) using this link: [Wolbanking77](https://kaggle.com/datasets/6f4251e190df4bb2c531856486d30b80c619155d2906f8fb3cd4448477a901b9)

Copy the text directory to the following directory : ```dataset/```

Copy the audio dataset to the following directory : ```dataset/```


# Requirements
## NLP
### Baselines

To install requirements for baseline models (KNN, SVM, Logistic Regression, Naive Bayes, LASER+MLP, LASER+CNN):

```setup
pip install -r tasks/nlp/ml_baselines_script/requirements_baseline.txt
```


### BertBase

To install requirements for BertBase :

```setup
sh setup_bert_base.sh
```

### AfroXLMR

To install requirements for AfroXLMR :

```setup
sh setup_afroxlrm.sh
```

### AfroLM

To install requirements for AfroLM :

```setup
sh setup_afrolm.sh
```

### mDeBERTa-v3

To install requirements for mDeBERTa-v3 :

```setup
sh setup_mdebertav3.sh
```

### AfritevaV2

To install requirements for AfritevaV2 :

```setup
sh setup_afritevav2.sh
```

### Llama-3.2

To install requirements for Llama-3.2 :

```setup
sh setup_llama3.2.sh
```

## ASR

To install requirements for Canary-1b-flash :

```setup
sh setup_canary1b_asr.sh
```

To install requirements for Phi-4-multimodal-instruct :

```setup
sh setup_phi4_asr.sh
```

To install requirements for Distil-whisper-large-v3.5 :

```setup
sh setup_whisper_asr.sh
```

# Training & evaluation

To run & evaluate the Baseline models in the paper, run this command:

## Baselines
```Run
python tasks/nlp/ml_baselines_script/run_nlp_baseline_benchmark.py dataset/text/
```

To train & evaluate the NLP models in the paper, run this command:
>📋  You can sprecify the corresponding split to run (5k_split or full).

## BertBase
```Train
python tasks/nlp/finetune/train_bert.py dataset/text/ 5k_split
```

```Evaluate
python tasks/nlp/finetune/eval_bert.py dataset/text/ 5k_split
```

>📋 You can use the same process as BertBase to train and evaluate AfroXLMR, AfroLM, mDeBERTa-v3 and AfritevaV2. All training and evaluation scripts are in the folder ```tasks/nlp/finetune```.

## Llama3.2

You can train and evaluate Llama3.2 (version 1B & 3B) using this following commands :
```Train
# data preprocessing
python tasks/nlp/finetune/dataset_preprocess_llama3.2-1B.py dataset/text/ 5k_split

# download Llama-3.2-1B-Instruct checkpoints by specifying your hugginface token.
tune download "meta-llama/Llama-3.2-1B-Instruct"  \   
    --output-dir "./Llama-3.2-1B-Instruct"  \   
    --hf-token "hugginface token"  \   
    --ignore-patterns "[]"

# start training process
tune run lora_finetune_single_device --config "custom_config.yaml" epochs=20

# run the evaluation script
python tasks/nlp/finetune/eval_llama3.2.py dataset/text/ 5k_split
```

# Project tree

```
📦checkpoint - this folder contains any saved model checkpoints.
📦config - contains canary flash config.
📦dataset
 ┣ 📂audio - here's the folder that contains all audios and transcriptions.
 ┃ 📂text  - This folder contains WolBanking77 text data.
📦results  - This folder contains results from benchmarks.
📦scripts  - Contains scripts dowloaded from Nvidia Nemo.
📦notebooks	- this folder contains any notebooks of the project.
📦lexicons	- this folder contains any texts and phonetic transcriptions for the audio dataset.
```


# License

>📋 Dataset and code are distributed under the CC-BY-4.0 license.
