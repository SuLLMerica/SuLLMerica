import numpy as np
import os
import traceback
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def search_faiss_index(faiss_index, query_embedding, k=5):
    # Validate input parameters
    if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a 1D numpy array")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    # Reshape the query embedding for FAISS (FAISS expects a 2D array)
    query_embedding_reshaped = query_embedding.reshape(1, -1)

    # Perform the search
    D, I = faiss_index.search(query_embedding_reshaped, k)

    # Return the indices and distances of the nearest neighbors
    return I, D



def get_query_embedding_BAAI(text):
        model_name = "BAAI/bge-large-en"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        query_embedding = hf.embed_query(text)
        #query_embedding = query_embedding.da
        #return response.data[0].embedding
        return np.array(query_embedding, dtype=np.float32)
             
def find_nearest_neighbors_faiss(query_text, embedding_query, faiss_index, data_mapping, k, source_mapping,  context=None):
    
    try:
        #query_embedding = get_query_embedding_OpenAILarge(query_text, context)
        #query_embedding = get_query_embedding_BAAI(query_text)
        query_embedding = np.array(embedding_query, dtype=np.float32)
        I, D = search_faiss_index(faiss_index, query_embedding, k)

        nearest_neighbors = []
        for index in I[0]:  
            if index < len(data_mapping):  
                data = data_mapping.get(index, "Data not found")
                source = source_mapping.get(index, "Source not found")
                nearest_neighbors.append((index, data, source))
        return nearest_neighbors
    except Exception as e:
        print(f"Error in find_nearest_neighbors_faiss: {str(e)}")
        traceback.print_exc()
        return []
 
