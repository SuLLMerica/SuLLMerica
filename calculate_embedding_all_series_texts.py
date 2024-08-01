import os
import json
from Source.input import get_documents
from Source.chunking import chunk_doc
from Source.calculate_embeddings import calculate_embedding
from Source.maneger_dataset import upload_dataset
import numpy as np

# Define the chunk size for document segmentation
CHUNK_SIZE = 250
NUM_CLUSTERS = 18

# Define the document series range
doc_series = np.arange(21, 39)
print(f"doc_series: {doc_series}")

# Retrieve the documents corresponding to the defined series
document_ds = get_documents(doc_series)
print(f"total_documents:{len(document_ds)}")

# Chunk documents based on provided chunk size and overlap
Document_ds = [chunk_doc(doc,CHUNK_SIZE) for doc in document_ds]
series_doc = {'Summaries':[]}
for series_number in doc_series:
    series_doc[f'Series{series_number}'] = []
    for doc in Document_ds:
        if doc[0]['source'][:2].isnumeric():
            if int(doc[0]['source'][:2]) == series_number:
                series_doc[f'Series{series_number}'].append(doc)
        else:
            if doc not in series_doc['Summaries']:
                series_doc['Summaries'].append(doc)
# Calculate embeddings for the chunked documents
series_docs = calculate_embedding(series_doc)

#Save embeddings docs
with open(f'series_embeddings_{CHUNK_SIZE}_chunksize_{NUM_CLUSTERS}numClusters.json', 'w') as arquivo_json:
    json.dump(series_docs, arquivo_json, indent=4)


# Upload the dataset to the SQLite database
dataset_path = f"series_embeddings_{CHUNK_SIZE}chunksize_{NUM_CLUSTERS}numClusters.json"
database = "./series_embeddings_{CHUNK_SIZE}chunksize_{NUM_CLUSTERS}numClusters.db"
upload_dataset(dataset_path, NUM_CLUSTERS,CHUNK_SIZE, series_docs=series_docs)

