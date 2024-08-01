
from Source.retrieval import find_nearest_neighbors_faiss
from Source.index import get_faiss_batch_index
from Source.get_definitions import define_TA_question
from Source.maneger_dataset import get_centroids_embeddings, get_embeddings_by_labels
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import traceback
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import functools



model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    
)


class GaussianInitialized(nn.Module):
    def __init__(self, in_features, out_features, mean, std):
        super(GaussianInitialized, self).__init__()
        self.mean = mean
        self.std = std
        self.w = torch.normal(self.mean, self.std, size=(in_features, out_features))

    def forward(self):
        return self.w


class CustomModel(nn.Module):
    def __init__(self, num_cluster, random_state=None):
        super(CustomModel, self).__init__()
        np.random.seed(random_state)
        self.device = torch.device("cuda")
        # self.layer1_1 = nn.Linear(1024, 256)
        self.layer1_1 = GaussianInitialized(1024, 256, mean=0, std=3)
        self.dropout1 = nn.Dropout(0.1)

        # self.layer2_1 = nn.Linear(num_cluster, 256)
        self.layer2_1 = GaussianInitialized(num_cluster, 256, mean=0, std=3).to(
            self.device
        )
        self.dropout2 = nn.Dropout(0.05)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.alfa = nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

        self.output_layer1 = nn.Linear(256, 128)
        self.output_layer2 = nn.Linear(128, num_cluster)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_1, input_2):

        x1 = torch.matmul(input_1, self.layer1_1().to(self.device))
        x1 = self.dropout1(x1)
        x1 = self.batchnorm1(x1)

        x2 = torch.matmul(input_2, self.layer2_1().to(self.device))
        x2 = self.dropout2(x2)
        x2 = self.batchnorm2(x2)

        weighted_x1 = self.alfa * x1
        weighted_x2 = self.beta * x2

        combined = weighted_x1 + weighted_x2

        output = self.output_layer1(combined)
        output = self.output_layer2(output)
        output = self.softmax(output)

        return output


class Query:
    def __init__(self, query, context):
        self.id = id
        self.question = query
        self.query = query
        self.enhanced_query = query
        self.embedding_query = []
        self.con_counter = {}
        self.topic_distr = []
        if isinstance(context, str):
            context = [context]
        self.context = context
        self.answer = ""
        self.rowcontext = []
        self.context_source = []
        self.possible_sources = []
        self.wg = []
        self.source_hit = {}
        self.document_accuracy = None
    

    def def_TA_question(self):
        self.query = define_TA_question(self.query)
        self.enhanced_query = self.query

    


    def get_embeddings_list(text_list):
        
        embeddings = []
        for text in text_list:
            response = hf.embed_query(text)
            embeddings.append(response)
        # for i in range(len(response.data)):
        #    embeddings.append(response.data[i].embedding)

        text_embeddings = {}
        # print(len(text_list))
        # print(len(embeddings))
        for index in range(len(text_list)):
            text_embeddings[text_list[index]] = embeddings[index]
        return text_embeddings

    def inner_product(a, b):  # É uma media eficaz
        """Compute the inner product of two lists."""
        return sum(x * y for x, y in zip(a, b))

    def get_cluster_chuncks(embeddings_list, num_cluster, conn=None):
        idx_cluster, cluster = get_centroids_embeddings(
            conn=conn
        )
        similarity_coloumn = []
        for embeddings in embeddings_list:
            coef = []
            for idx in idx_cluster:
                coef.append(Query.inner_product(embeddings, cluster[idx]))
            similarity_coloumn.append(coef)

        return similarity_coloumn

    def preprocessing_softmax(embeddings_list, num_cluster, conn=None):
        embeddings = np.array(embeddings_list)
        # similarity = np.array(Query.get_col2(embeddings))
        similarity = np.array(Query.get_cluster_chuncks(embeddings, num_cluster, conn=conn))

        X_train_1_tensor = torch.tensor(embeddings, dtype=torch.float32, device="cuda")

        _similarity = torch.from_numpy(similarity)
        _similarity = torch.tensor(similarity, dtype=torch.float32, device="cuda")
        X_train_2_tensor = torch.nn.functional.softmax(10 * _similarity, dim=-1)

        dataset = TensorDataset(X_train_1_tensor, X_train_2_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

        return dataloader


    def get_embeddings(text):
        response = hf.embed_query(text)
        # return response.data[0].embedding
        return response


    @functools.lru_cache
    def predict_wg(self, num_cluster, top_k_clusters, conn=None):
        model = CustomModel(num_cluster, random_state=42)
        # model.load_state_dict(torch.load('router_new.pth', map_location='cuda'))

        device = torch.device("cuda")
        model.to(device)
        model.eval()
        text_list = []
        text_embeddings = Query.get_embeddings_list([self.enhanced_query])
        cluster_list = []
        self.embedding_query = text_embeddings[self.enhanced_query]
        test_dataloader = Query.preprocessing_softmax(
            [self.embedding_query], num_cluster, conn=conn
        )
        with torch.no_grad():
            for X1, X2 in test_dataloader:
                # Move data to the same device as the model
                X1, X2 = X1.to(device), X2.to(device)
                outputs = model(X1, X2)
                top_values, top_indices = outputs.topk(top_k_clusters, dim=1)
                # Convert the indices to a numpy array
                predicted_indices = top_indices.cpu().numpy()
        self.wg = predicted_indices.tolist()
        # print(self.wg)


    def get_question_context_faiss(self, batch, k, use_context=False):
        try:

            faiss_index, faiss_index_to_data_mapping, source_mapping = (
                get_faiss_batch_index(batch)
            )

            # print(f"FAISS INDEX: {faiss_index}")
            if use_context:
                result = find_nearest_neighbors_faiss(
                    self.query,
                    self.embedding_query,
                    faiss_index,
                    faiss_index_to_data_mapping,
                    k,
                    source_mapping=source_mapping,
                    context=self.context,
                )
            else:
                result = find_nearest_neighbors_faiss(
                    self.query,
                    self.embedding_query,
                    faiss_index,
                    faiss_index_to_data_mapping,
                    k,
                    source_mapping=source_mapping,
                )

            if isinstance(result, list):
                self.context = []
                self.context_source = []
                for i in range(len(result)):
                    index, data, source = result[i]
                    self.context.append(
                        f"\nRetrieval {i+1}:\n...{data}...\nThis retrieval is performed from the document {source}.\n"
                    )
                    self.context_source.append(f"Index: {index}, Source: {source}")
            else:
                self.context = result
        except Exception as e:
            print(f"An error occurred while getting question context: {e}")
            print(traceback.format_exc())
            self.context = "Error in processing"
