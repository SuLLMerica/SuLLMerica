import os
import traceback
from Source.embeddings import get_embeddings
from Source.query import Query
import sys
import traceback
from Source.input import get_documents
from Source.chunking import chunk_doc
from Source.calculate_embeddings import calculate_embedding
from Source.maneger_dataset import get_embeddings_by_labels
from Source.generate_question import generate_questions_training
import git
import random
import ujson as json
import logging
import pandas as pd

# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# logging.getLogger().setLevel(logging.ERROR)

NUM_CLUSTERS = 18
TOP_K_CLUSTERS = 8
TOP_K_CHUNKS = 5


def choose_random_question(data):
    while True:
        random_question = random.choice(list(data.values()))

        if "3GPP" in random_question["question"]:
            return random_question["question"]
        else:
            continue


def TelcoRAG(query, options):
    try:
        
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        question = Query(query, options, [])

        question.def_TA_question()

        question.predict_wg(NUM_CLUSTERS, TOP_K_CLUSTERS)

        embeddings_per_cluster = get_embeddings_by_labels(
            "Data/cluster_data_BisectingKMeans_18_250_chunksize.db", question.wg
        )

        question.get_question_context_faiss(
            batch=embeddings_per_cluster, k=TOP_K_CHUNKS, use_context=False
        )

        question.candidate_answers_phi_model()
        response = question.answer
        context = question.context

        return response, context, query

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":

    questions = generate_questions_training("TeleQnA.json")
    code_question = []
    correct_answer = []
    predicted_answer = []
    # hits =0
    # question = choose_random_question(questions)
    for single_question in questions:
        code_question.append(single_question["code"])
        question = single_question["question"]
        options = single_question["options"]
        correct_answer.append(single_question["answer"])
        try:
            response = TelcoRAG(question, options)
            print(
                f"""Generated response to the question:
                          {response[2]}
                Is:
                           {response[0]} """
            )
        except Exception as e:
            print("Encountered an error and moving to the next case.")
            print(traceback.format_exc())
        if len(response[0]) >= 2:
            predicted_answer.append(response[0][-1])
        else:
            predicted_answer.append(None)

    data = {
        "Code question": code_question,
        "Correct answer": correct_answer,
        "Predicted answer": predicted_answer,
    }
    df = pd.DataFrame(data)
    df.to_csv(
        f"results_gaussian_{NUM_CLUSTERS}_{TOP_K_CLUSTERS}_{TOP_K_CHUNKS}.csv",
        index=False,
    )
    # print dataframe
    print(df.head())
    df["Correct answer"] = df["Correct answer"].astype(int)
    df["Predicted answer"] = pd.to_numeric(df["Predicted answer"], errors="coerce")
    df["Correct"] = df["Correct answer"] == df["Predicted answer"]
    accuracy = df["Correct"].mean() * 100
    print(f"accuracy: {accuracy:.2f}%")
