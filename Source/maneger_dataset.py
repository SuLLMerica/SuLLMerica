import ujson as json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sqlite3
from tqdm import tqdm
import time


def upload_dataset(dataset_path, num_clusters, CHUNK_SIZE,series_docs=None):
    # Carregar embeddings do arquivo JSON
    if(series_docs):
        data = series_docs
    else:
        print(f"Loading dataset from {dataset_path}...")
        with open(dataset_path, "r") as file:
            data = json.load(file)

    # Extrair os embeddings
    embeddings = []
    text = []
    source = []

    pb = tqdm(data.items(), total=len(data), desc="creating embeddings array", unit="series")

    for series, doc_list in pb:
        for doc in doc_list:
            for id, single_doc in enumerate(doc):
                embeddings.append(single_doc["embedding"])
                source.append(single_doc["source"])
                text.append(single_doc["text"])

    embeddings = np.array(embeddings)
    # print(embeddings.shape)

    # Aplicar K-means
    print(f"Applying K-means with {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)


    # Obter os rótulos dos clusters
    labels = kmeans.labels_

    # Obter os centroides dos clusters
    centroids = kmeans.cluster_centers_

    # Conectar ao banco de dados SQLite (ou criar um novo)
    conn = sqlite3.connect(f"cluster_data_{num_clusters}_{CHUNK_SIZE}.db")
    cursor = conn.cursor()

    # Criar tabelas
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS points (
            id INTEGER PRIMARY KEY,
            embedding TEXT,
            text TEXT,
            source TEXT,
            label INTEGER
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS centroids (
            id INTEGER PRIMARY KEY,
            cluster INTEGER,
            centroid TEXT
        )
    """
    )

    # Inserir dados dos pontos
    pb = tqdm(enumerate(embeddings), total=len(embeddings), desc="Uploading points", unit="points")
    for i, embedding in pb:
        embedding_str = json.dumps(
            embedding.tolist()
        )  # Converter o array numpy para string JSON
        cursor.execute(
            """
            INSERT INTO points (embedding, text, source, label) VALUES (?, ?, ?, ?)
        """,
            (embedding_str, text[i], source[i], int(labels[i])),
        )

    # Inserir dados dos centroides

    pb = tqdm(enumerate(centroids), total=len(centroids), desc="Uploading centroids", unit="centroids")

    for i, centroid in pb:
        centroid_str = json.dumps(
            centroid.tolist()
        )  # Converter o array numpy para string JSON
        cursor.execute(
            """
            INSERT INTO centroids (cluster, centroid) VALUES (?, ?)
        """,
            (i, centroid_str),
        )

    # Commit das mudanças e fechar a conexão
    conn.commit()
    conn.close()


def get_embeddings_by_labels(label_groups,conn=None):
    all_data = []

    # Conectar ao banco de dados SQLite
    cursor = conn.cursor()


    for labels in label_groups:
        # Construir a consulta SQL com os labels fornecidos
        placeholders = ",".join("?" for _ in labels)
        query = f"SELECT embedding, text, source FROM points WHERE label IN ({placeholders})"

        # Executar a consulta SQL
        cursor.execute(query, labels)


        # Extrair os resultados
        rows = cursor.fetchall()

        
        # Converter as strings JSON de volta para arrays numpy
        data_group = []
        for row in rows:
            embedding = np.array(json.loads(row[0]))
            data_group.append({"embedding": embedding, "text": row[1], "source": row[2]})

        
        all_data.append(data_group)



    return all_data


def get_centroids_embeddings(conn=None):

    cursor = conn.cursor()
    # Consultar todos os centroides
    cursor.execute("SELECT cluster, centroid FROM centroids")
    centroids = cursor.fetchall()

    cluster_id = []
    centroid_list = []
    # Exibir os centroides
    # print("\nCentroides dos Clusters:")
    for single_centroid in centroids:
        cluster, centroid_str = single_centroid
        single_centroid = json.loads(
            centroid_str
        )  # Converter string JSON de volta para lista
        cluster_id.append(cluster)
        centroid_list.append(single_centroid)
        # print(f'Cluster: {cluster}, Centroid: {len(single_centroid)} shape')


    return cluster_id, centroid_list



