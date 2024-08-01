import faiss
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


os.environ['OMP_NUM_THREADS'] = '8'

def create_faiss_index_IndexFlatIP(embeddings, data, source):
    """Create FAISS IndexFlatIP from embeddings and maps indices to data and source."""
    try:
        #logging.info("Creating IndexFlatIP...")
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        index_to_data_mapping = {i: data[i] for i in range(len(data))}
        index_to_source_mapping = {i: source[i] for i in range(len(source))}
        return index, index_to_data_mapping, index_to_source_mapping
    except Exception as e:
        logging.error(f"Error creating FAISS index: {e}")
        return None, None, None

def get_faiss_batch_index(embedded_batch):
    """Generate FAISS index from a batch of embeddings, handling missing embeddings by generating them."""
    try:
        source = [chunk['source'] for chunked_batch in embedded_batch for chunk in chunked_batch]
        embeddings = []
        data = []

        for doc in embedded_batch:
            embeddings_batch = []
            for chunk in doc:
                if 'embedding' in chunk:
                    embeddings_batch.append(chunk['embedding'])
                else:
                    logging.error(f"Missing embedding for chunk: {chunk['text']}")

            embeddings.extend(embeddings_batch)
            data.extend([chunk['text'] for chunk in doc])

        embeddings = np.array(embeddings, dtype=np.float32)
        return create_faiss_index_IndexFlatIP(embeddings, data, source)
    except Exception as e:
        logging.error(f"Failed to process batch for FAISS indexing: {e}")
        return None, None, None

