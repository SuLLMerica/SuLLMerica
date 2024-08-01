from Generate_RAG_contexts import Get_RAG_Context
import re
import ujson as json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import warnings
import logging

# Set the logging level to WARNING for the entire application
# This suppresses INFO and DEBUG messages
logging.basicConfig(level=logging.WARNING)

# Suppress specific FutureWarning from huggingface_hub
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated and will be removed in version 1.0.0",
    category=FutureWarning,
    module="huggingface_hub",
)

# Suppress specific UserWarning about flash attention
warnings.filterwarnings(
    "ignore",
    message="Torch was not compiled with flash attention",
    category=UserWarning,
    module="transformers.models.phi.modeling_phi",
)


def generate_questions_test(path):
    with open(path) as json_file:
        docs = json.load(json_file)
    questions = []
    for key, value in docs.items():
        question = {}
        question["code"] = key.split(" ")[1]
        question["question"] = value["question"]

        # Modificação para que options seja um dicionário
        options = {}
        for code, option in value.items():
            if "option" in code:
                options[code] = option
        question["options"] = options

        questions.append(question)

    return questions


def concatenate_testing_questions(paths):
    questions = []
    for path in paths:
        questions += generate_questions_test(path)
    return questions


if __name__ == "__main__":

    # Set the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and tokenizer
    model_name = "claudiomello/phi2-SFT-TeleQnA-ContextTrained"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # Get the testing data
    test_data = ["Data/test/TeleQnA_testing1.txt", "Data/test/questions_new.txt"]
    questions = concatenate_testing_questions(test_data)

    # Create a df to store the questions and answers
    df = pd.DataFrame(columns=["Question_ID", "Answer_ID"])

    pb = tqdm(
        questions, total=len(questions), desc="Generating test results", unit="question"
    )

    fault_answers = 0

    # Generate the test results
    for question in pb:

        # Get the query and answer
        query = str(question["question"])
        try:
            option_1 = str(question["options"]["option 1"])
            option_1_exists = True
        except KeyError:
            option_1 = ""
            option_1_exists = False
        try:
            option_2 = str(question["options"]["option 2"])
            option_2_exists = True
        except KeyError:
            option_2 = ""
            option_2_exists = False
        try:
            option_3 = str(question["options"]["option 3"])
            option_3_exists = True
        except KeyError:
            option_3 = ""
            option_3_exists = False
        try:
            option_4 = str(question["options"]["option 4"])
            option_4_exists = True
        except KeyError:
            option_4 = ""
            option_4_exists = False
        try:
            option_5 = str(question["options"]["option 5"])
            option_5_exists = True
        except KeyError:
            option_5_exists = False
            option_5 = ""

        # Update the question and answer in the DataFrame
        merged_question = (
            (
                query
                + "\n"
                + ("\n1. " + option_1 if option_1_exists else "")
                + ("\n2. " + option_2 if option_2_exists else "")
                + ("\n3. " + option_3 if option_3_exists else "")
                + ("\n4. " + option_4 if option_4_exists else "")
                + ("\n5. " + option_5 if option_5_exists else "")
            )
            + "\n"
            + "Choose the correct option from the above options:"
        )

        # Get the Context
        pb.write("Getting the context")
        context = Get_RAG_Context(question["question"])[0]
        pb.write("Context obtained")

        full_query = f"""Please provide the answer to the following multiple choice question: {merged_question}

        Considering the following context: {str(context)}

        Please provide the answer to the the following multiple choice question: {merged_question}

        Write only the option number corresponding to the correct answer."""

        # Create a dialogue with the user and assistant messages
        inputs = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": full_query,
                },
            ],
            return_tensors="pt",
        )

        # Generate the response
        response = model.generate(
            inputs.to(device),
            max_length=2048,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode the response
        response = tokenizer.decode(response[0], skip_special_tokens=True)

        # Extract the answer from the response
        match = re.search(r"<\|im_start\|>assistant\n(\d+)", response)
        if match:
            answer = int(match.group(1))
        else:
            answer = -1
            fault_answers += 1
            pb.set_postfix(fault_answers=fault_answers)

        # Update the question and answer in the DataFrame
        df = df._append(
            {
                "Question_ID": question["code"],
                "Answer_ID": answer,
                "Response": response,
                "Context": str(context),
                "Merged_Question": merged_question,
            },
            ignore_index=True,
        )

    # Save the results to a CSV file
    df.to_csv("test_results_with_context.csv", index=False)

    pb.write("Test results saved to 'test_results_with_context.csv'")
