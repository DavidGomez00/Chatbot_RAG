import json
import os

import chromadb
import sentence_transformers
from chromadb.api.types import EmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
CHUNKS_PATH = os.path.join(os.getenv("DATA_PATH"), 'chunks/')
if not os.path.exists(CHUNKS_PATH):
    os.makedirs(CHUNKS_PATH)

embedding_model_dict = {"multilinguale5": 'intfloat/multilingual-e5-large',
                        "BGE-M3": "BAAI/bge-m3",
                       }

# Define a custom embedding function
class MyEmbeddingFunction(EmbeddingFunction):
    ''' Custom embedding function that uses a SentenceTransformer model to encode text into embeddings
    '''
    def __init__(self, model_name):
        super(MyEmbeddingFunction).__init__()
        self.model = sentence_transformers.SentenceTransformer(model_name, trust_remote_code=True)
    def __call__(self, texts):
        return self.model.encode(texts).tolist()


def create_collection(model:str, strategy:str="nat"):
    # Check if the model is valid
    if model not in embedding_model_dict:
        raise ValueError(f"Invalid model {model}. Valid models are: {', '.join(embedding_model_dict.keys())}")

    embedding_function = MyEmbeddingFunction(embedding_model_dict[model])
    chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH"))
    collection = chroma_client.get_or_create_collection(name="RAG_VIH_"+model,
                                                        embedding_function=embedding_function)
    if collection.count() > 0:
        return collection
    
    chunks = []
    for file in os.listdir(os.path.join(CHUNKS_PATH, strategy)):
        with open(os.path.join(os.path.join(CHUNKS_PATH, strategy), file), 'r') as f:
            chunks += json.load(f)

    collection.add(ids=[str(chunk["chunk_id"]) for chunk in chunks],
                    documents=[chunk["text"] for chunk in chunks],
                    metadatas=[{"section_id": chunk["section_id"],
                                "document_id": chunk["document_id"]} for chunk in chunks])
    return collection
    

def query(model:str, query:list, n_results:int):
    
    collection = create_collection(model)
    results = []
    for q in query:
        results.append(collection.query(query_texts=q, n_results=n_results))

    return results

