import os
import traceback
from Source.query import Query
from Source.maneger_dataset import get_embeddings_by_labels



def Get_RAG_Context(query, conn, NUM_CLUSTERS, TOP_K_CLUSTERS, TOP_K_CHUNCKS):
    """
    Generates RAG (Retrieval-Augmented Generation) context for a given query.

    Args:
        query (str): The main question or query.
        options (dict): A dictionary containing the answer options for the question.
        conn (sqlite3.Connection, optional): Connection to the SQLite database. Defaults to None.

    Returns:
        str: The generated RAG context.

    Raises:
        Exception: If an error occurs during the generation of the RAG context.
    """
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        question = Query(query, [])

        question.def_TA_question()

        question.predict_wg(NUM_CLUSTERS, TOP_K_CLUSTERS, conn=conn)

        embeddings_per_cluster = get_embeddings_by_labels(
            question.wg, conn=conn
        )

        question.get_question_context_faiss(
            batch=embeddings_per_cluster, k=TOP_K_CHUNCKS, use_context=False
        )

        return question.context

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
