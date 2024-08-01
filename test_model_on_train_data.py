import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import ujson as json

TEST_DATA_PATH = "./TeleQnA_Train_With_RAG_Context_final.json"
MODEL_NAME = "claudiomello/Phi-2-SFT-Context-Generic"


if __name__ == "__main__":

    # Set the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)

    # Get the testing data
    with open(TEST_DATA_PATH) as json_file:
        questions = json.load(json_file)

correct_answers = 0
answers = 0

pb = tqdm(questions, desc="Generating test answers", total=len(questions), unit="Question")

# Iterate over the rows of the DataFrame
for question_iter in tqdm(questions):
    # Get the question and answer
    question = str(question_iter["question"])

    try:
        option_1 = str(question_iter["options"]["option 1"])
        option_1_exists = True
    except KeyError:
        option_1 = ""
        option_1_exists = False
    try:
        option_2 = str(question_iter["options"]["option 2"])
        option_2_exists = True
    except KeyError:
        option_2 = ""
        option_2_exists = False
    try:
        option_3 = str(question_iter["options"]["option 3"])
        option_3_exists = True
    except KeyError:
        option_3 = ""
        option_3_exists = False
    try:
        option_4 = str(question_iter["options"]["option 4"])
        option_4_exists = True
    except KeyError:
        option_4 = ""
        option_4_exists = False
    try:
        option_5 = str(question_iter["options"]["option 5"])
        option_5_exists = True
    except KeyError:
        option_5_exists = False
        option_5 = ""

    # Update the question and answer in the DataFrame
    merged_question = (
        (
            question
            + "\n"
            + ("\n1. " + option_1 if option_1_exists else "")
            + ("\n2. " + option_2 if option_2_exists else "")
            + ("\n3. " + option_3 if option_3_exists else "")
            + ("\n4. " + option_4 if option_4_exists else "")
            + ("\n5. " + option_5 if option_5_exists else "")
        )
        + "\n\n"
        + "Choose the correct option from the above options"
    )

    context = ""
    for ret in question_iter["context"]:
        context += ret

    full_context = (
        f"Considering the following context:\n{str(context)}\n"
        + (
            f"Terms and Definitions:\n{question_iter['terms']}\n"
            if question_iter["terms"]
            else ""
        )
        + (
            f"Abbreviations: {question_iter['abbreviations']}\n"
            if question_iter["abbreviations"]
            else ""
        )
    )

    full_question = (
        f"Please provide the answer to the the following multiple choice question:\n{merged_question}\n"
        + "Write only the option number corresponding to the correct answer."
    )

    input_tensor = tokenizer.apply_chat_template(
        [
            {
                "role": "context",
                "content": full_context,
            },
            {
                "role": "user",
                "content": full_question,
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



    # Extract the answer from the full answer
    match = re.search(r"The correct option from the above is number (\d)", response)
    if match:
        answer = match.group(1)
    else:
        print("Answer not found")

    


    answers += 1
    if int(answer) == int(question_iter["answer"][7]):
        correct_answers += 1

    pb.set_postfix({"Accuracy": correct_answers / answers*100})




    

