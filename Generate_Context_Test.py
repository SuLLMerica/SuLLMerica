import os
import traceback
from Source.query import Query
import traceback
from Source.maneger_dataset import get_embeddings_by_labels
from Source.generate_question import generate_questions_training
from Source.get_RAG_context import Get_RAG_Context
import ujson as json
from tqdm import tqdm
import sqlite3
import concurrent.futures



# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# logging.getLogger().setLevel(logging.ERROR)

NUM_CLUSTERS = 18
TOP_K_CLUSTERS = 8
TOP_K_CHUNCKS = 5
DATABASE_PATH="cluster_data_BisectingKMeans_18_250_chunksize.db"
TEST_FILES=["./Data/test/questions_new.txt", "./Data/test/TeleQnA_testing1.txt"]




if __name__ == "__main__":
    import pandas as pd
    
    # Get the training data
    df = pd.read_csv("./terms_abbreviations_full_test.csv")

    # Add a column for the RAG context
    df["context"] = None

    pb = tqdm(
        df.iterrows(),
        total=len(df),
        desc="Generating RAG Contexts",
        unit="question",
    )

    questions = []

    conn = sqlite3.connect(DATABASE_PATH)

    for index, row in pb:
        question = row["question"]
        options = {}

        try:
            option_1 = str(row["option 1"])
            if option_1 != "nan" and option_1 != "":
                options["option 1"] = option_1
        except KeyError:
            pass

        try:
            option_2 = str(row["option 2"])
            if option_2 != "nan" and option_2 != "":
                options["option 2"] = option_2
        except KeyError:
            pass

        try:
            option_3 = str(row["option 3"])
            if option_3 != "nan" and option_3 != "":
                options["option 3"] = option_3
        except KeyError:
            pass

        try:
            option_4 = str(row["option 4"])
            if option_4 != "nan" and option_4 != "":
                options["option 4"] = option_4
        except KeyError:
            pass
            
        try:
            option_5 = str(row["option 5"])
            if option_5 != "nan" and option_5 != "":
                options["option 5"] = option_5
        except KeyError:
            pass

        terms = None
        if str(row["terms"]) != "nan" and row["terms"] != "":
            terms = row["terms"]
        
        abbreviations = None
        if str(row["abbreviation"]) != "nan" and row["abbreviation"] != "":
            abbreviations = row["abbreviation"]
        
        
        try:
            context = Get_RAG_Context(question, terms, abbreviations, conn=conn)
            df.at[index, "context"] = context
        except Exception as e:
            print(f"An error occurred: {e}")
            print(traceback.format_exc())


        obj = {
            "question": question,
            "options": options,
            "context": context,
            "terms": terms,
            "abbreviations": abbreviations,
            "category": row["category"],
            "Question_ID": row["question_id"]
        }
        questions.append(obj)

        

    # Fechar a conexão
    conn.close()


    df.to_csv("TeleQnA_Test_With_RAG_Context_final.csv", index=False)

    # Save questions as JSON

    with open("TeleQnA_Test_With_RAG_Context_final.json", "w") as f:
        json.dump(questions, f)
