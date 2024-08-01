import logging
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from tqdm import tqdm


def calculate_embedding(series_docs):
    """
    Calculate and add embeddings to each chunk of documents.
    
    Args:
        series_docs (dict): Dictionary containing series documents and their respective chunks.
        
    Returns:
        dict: Updated dictionary with embeddings added to each document chunk.
    """
    # Define model parameters
    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": True}

    # Initialize the HuggingFace embeddings model
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    pb = tqdm(series_docs.items(),total=len(series_docs), desc="Calculating embeddings series_docs", unit="series_docs")
    

    # Iterate over each document series and its chunks
    for doc_key, doc_chunks in pb:
        pb2 = tqdm(enumerate(doc_chunks), total=len(doc_chunks), desc="Calculating embeddings docs", unit="docs", leave=False)
        for idx_doc,chunk in pb2:
            updated_chunks = []
            pb3 = tqdm(enumerate(chunk), total=len(chunk), desc="Calculating embeddings chunks", unit="chunks", leave=False)
            for idx, single_chunk in pb3:
                try:
                    # Calculate embeddings for the chunk text
                    embeddings = hf.embed_query(single_chunk['text'])
                    single_chunk['embedding'] = embeddings
                    updated_chunks.append(single_chunk)
                except IndexError:
                    logging.warning(f"Embedding index {idx} out of range for {doc_key}.")
                except Exception as e:
                    logging.error(f"Error processing chunk {idx} for {doc_key}: {e}")
            # Update the series documents with the newly embedded chunks
            series_docs[doc_key][idx_doc] = updated_chunks

    return series_docs
