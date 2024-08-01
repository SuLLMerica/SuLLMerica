import pandas as pd
import ujson as json
import re
import os
import traceback
from Source.query import Query
from Source.maneger_dataset import get_embeddings_by_labels
from Source.generate_question import generate_questions_training
from Source.enhancement_query import EnhancementQuery
from Source.get_RAG_context import Get_RAG_Context
from tqdm import tqdm
import sqlite3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer





NUM_CLUSTERS = 18
TOP_K_CLUSTERS = 8
TOP_K_CHUNCKS = 5
DATABASE_PATH="cluster_data_BisectingKMeans_18_250_chunksize.db"
PATH_TERMS_FILE = "./Data/TermsAndDefinitions/terms_definitions.json"
PATH_ABBREVIATIONS_FILE = "./Data/TermsAndDefinitions/abbreviations_definitions.json"
MODEL_NAME = "claudiomello/Phi-2-TeleQnA-Finetune-Final"


# Create a class for enhancement
enhacenment_query = EnhancementQuery(file_name_terms=PATH_TERMS_FILE, file_name_abbreviations=PATH_ABBREVIATIONS_FILE)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
conn = sqlite3.connect(DATABASE_PATH)

while True:
    question = input("Please enter your question: ")

    print("Processing...")

    terms, abreviations = enhacenment_query.define_TA_question(question)

    try:
        context_array = Get_RAG_Context(question, conn, NUM_CLUSTERS, TOP_K_CLUSTERS, TOP_K_CHUNCKS)
        context = ""
        for ret in context_array:
            context += ret
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())



    full_context = (
        f"Considering the following context:\n{str(context)}\n"
        + (
            f"Terms and Definitions:\n{terms}\n"
            if terms
            else ""
        )
        + (
            f"Abbreviations: \n{abreviations}\n"
            if abreviations
            else ""
        )
    )

    input_tensor = tokenizer.apply_chat_template(
        [
            {
                "role": "context",
                "content": full_context,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        return_tensors="pt",
    )

    # Generate the answer
    with torch.no_grad():
        output = model.generate(
            input_tensor.to(device),
            max_length=2048,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode the answer
    response = tokenizer.decode(output[0], skip_special_tokens=True)


    try:
        initial_index = response.find("<|im_start|>assistant")
        final_index = response[initial_index:].find("<|im_end|>")
        correction = len("<|im_start|>assistant")

        answer = response[initial_index + correction:final_index+initial_index]
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())





