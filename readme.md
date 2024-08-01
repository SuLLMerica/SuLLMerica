- [SuLLMerica-ITU](#sullmerica-itu)
  - [Description](#description)
  - [Code pipeline](#code-pipeline)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Generating the cluster of embeddings for the RAG](#generating-the-cluster-of-embeddings-for-the-rag)
    - [Generating the training data for the finetuning](#generating-the-training-data-for-the-finetuning)
    - [Finetuning the model](#finetuning-the-model)
      - [Parameters](#parameters)
    - [Testing the model](#testing-the-model)
  - [Our data](#our-data)
  - [License](#license)
  - [Contact](#contact)


# SuLLMerica-ITU

This repository contains the code for the SuLLMerica project, developed for the [Specializing Large Language Models for Telecom Networks by ITU AI/ML in 5G Challenge](https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks) competition.





## Description

This is a RAG and finetuned model specialyzed for Question answering in the Telecomunications scope, it was made and trained in order to get the best performance in the [TeleQnA dataset](https://huggingface.co/datasets/netop/TeleQnA).

The proposed method is discussed in our [Article]() (Not yet released)

## Code pipeline
The code consists of a RAG system that retrieves context from the rel18 documents of the 3GPP Standart combined with a query enhancement process that extracts the terms, definitions and abreviations of the question and appends their meanings to the context as well as a LLM model that was finetuned from [phi-2](https://huggingface.co/microsoft/phi-2) with the context data to improve it`s utilization of the context.

## Installation

OBS: to run this code you must have a GPU that supports CUDA

To run this code locally, follow these steps:

1. Clone the repository: `git clone `
2. Install the requirements `pip install -r requirements.txt`
3. Donwload the data from the [kaggle dataset](https://www.kaggle.com/datasets/frankmorte/sullmerica-data)
    The code utilizes only the **cluster_data_BisectingKMeans_18_250_chunksize.db** out of the .db files so if you want to only use the final version you can ignore the other .db files.
4. Move these files to the main directory of the repository.


## Usage
All the data generated and the finetuned model are already available on the [Our data](#our-data) section.
If you wish only to test the model performance go to the [Testing the model](#testing-the-model) section.
Instead, if tou wish to verify our process or improve it, follow the instructions bellow

### Generating the cluster of embeddings for the RAG
1. On the [calculate_embedding_all_series_texts.py](calculate_embedding_all_series_texts.py) script define the **NUM_CLUSTERS** and **CHUNK_SIZE** variables (The proposed values are 18 and 250 respectively and are already defined on the file)
2. Run the [calculate_embedding_all_series_texts.py](calculate_embedding_all_series_texts.py) script to generate the files


### Generating the training data for the finetuning
1. If you generated your own database from the steps above change the **NUM_CLUSTERS**, **TOP_K_CLUSTERS**, **TOP_K_CHUNCKS** and **DATABASE_PATH** variables to the values you used, if not proceed to step 2
2. Run the [train_whole_process.ipynb](train_whole_process.ipynb) notebook to generate the dataset used for the finetuning

### Finetuning the model
The finetuned model used for the paper is [TeleQnA-Phi2-Phinetune](https://huggingface.co/SuLLMerica/TeleQnA-Phi2-Phinetune)
Our team used the [Huggingface autotrain](https://huggingface.co/autotrain) to train this model.

#### Parameters

    "model": "microsoft/phi-2",
    "add_eos_token": true,
    "block_size": 1024,
    "model_max_length": 2048,
    "padding": "right",
    "trainer": "default",
    "use_flash_attention_2": true,
    "log": "tensorboard",
    "disable_gradient_checkpointing": false,
    "logging_steps": 1,
    "eval_strategy": "epoch",
    "save_total_limit": 1,
    "auto_find_batch_size": true,
    "mixed_precision": "fp16",
    "lr": 3e-05,
    "epochs": 30,
    "batch_size": 2,
    "warmup_ratio": 0.1,
    "gradient_accumulation": 4,
    "optimizer": "adamw_torch",
    "scheduler": "linear",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "seed": 42,
    "chat_template": "none",
    "quantization": "int4",
    "target_modules": "all-linear",
    "merge_adapter": true,
    "peft": true,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "text_column": "text",
    "unsloth": false



### Testing the model
Because the test data was not published yet we don`t show the percentage score of the test data.
1. Run the [test_whole_process.ipynb](test_whole_process.ipynb) notebook to generate the results of the test data of the competition.


## Our data
- Aditional data that was too big for GitHub: [Aditional data](https://www.kaggle.com/datasets/frankmorte/sullmerica-data)
- Dataset used for the finetuning: [SuLLMerica/TeleQnA-prompt-with-context-phinetune](https://huggingface.co/datasets/SuLLMerica/TeleQnA-prompt-with-context-phinetune)
- Training questions with the generated RAG context: [SuLLMerica/TeleQnA_Train_With_RAG_Context](https://huggingface.co/datasets/SuLLMerica/TeleQnA_Train_With_RAG_Context)
- Test questions with the generated RAG context: [SuLLMerica/TeleQnA_Test_With_RAG_Context](https://huggingface.co/datasets/SuLLMerica/TeleQnA_Test_With_RAG_Context)
- Finetuned model: [SuLLMerica/TeleQnA-Phi2-Phinetune](https://huggingface.co/SuLLMerica/TeleQnA-Phi2-Phinetune)

## License

[Specify the license under which the code is distributed, if applicable]

## Contact

[Provide contact information for the project maintainers or team members]
