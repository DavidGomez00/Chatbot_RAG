import os
import json
import lancedb
from dotenv import load_dotenv

from cachetools import cached
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import TextEmbeddingFunction, EmbeddingFunctionRegistry

from lancedb.embeddings.registry import register
import requests

load_dotenv()

embedding_model_dict = {
    #"multilinguale5": 'zylonai/multilingual-e5-large',
    "BGE-M3": "bge-m3",
    "jina": "jina/jina-embeddings-v2-base-es",
}

## ici on va créer les embeddings vectors
class RemoteOllamaEmbeddingFunction(TextEmbeddingFunction):
    server_ip: str = None
    model_name: str = None
    _ndims: int = None 

    def __init__(self, server_ip: str = None, model_name: str = None):
        super().__init__()
        self.server_ip = server_ip
        self.model_name = model_name
        self._ndims = None
    
    def generate_embeddings(self, texts:list):
        '''Calls ollama server to generate embeddings for the given texts. If the texts
        list is too long, it will split it into batches to avoid overworking the server.
        '''
        
        _batch_size = 32
        if len(texts) > _batch_size:
            batches = [texts[i:i + _batch_size] for i in range(0, len(texts), _batch_size)]
            embeddings = []
            for i, batch in enumerate(batches):
                try:
                    print(f"Processing batch {i} of {len(batches)}")
                    response = requests.post(
                        url=f"http://{self.server_ip}/api/embed",
                        json={
                            "model": self.model_name,
                            "input": batch
                        }
                    )
                    response.raise_for_status()
                    embeddings.extend(response.json()["embeddings"])
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    print(f"Batch contains {len(batch)} elements:")
                    print(batch)
            return embeddings
        
        else:
            response = requests.post(
                url=f"http://{self.server_ip}/api/embed",
                json={
                    "model": self.model_name,
                    "input": texts
                }
            )
            response.raise_for_status()
            return response.json()["embeddings"]

    def ndims(self):
        if self._ndims is None:
            self._ndims = len(self.generate_embeddings(["foo"])[0])
        return self._ndims

    @cached(cache={})
    def _embedding_model(self):
        return self

@register("remote-ollama")
class RemoteOllamaEmbeddings(RemoteOllamaEmbeddingFunction):
    def __init__(self, model_name: str, max_retries: int = 10, server_ip: str = None):
        super().__init__(
            server_ip=os.getenv("OLLAMA_SERVER_IP"), 
            model_name=model_name
        )
        
def create_collection(models: list, strategies: list):
    '''Create a database with tables for each configuration of the model.'''

    db = lancedb.connect(os.getenv("LANCE_DB_PATH")) # à remplacer par qdrant
    registry = EmbeddingFunctionRegistry.get_instance()

    for model in models:
        embedder = registry.get("remote-ollama").create(model_name=embedding_model_dict[model])
        
        class Chunk(LanceModel):
            text: str = embedder.SourceField()
            vector: Vector(embedder.ndims()) = embedder.VectorField()  # type: ignore
            chunk_id: str
            section_id: str
            document_id: str

        for strategy in strategies:
            table_name = f"chunks_{model}_{strategy}"
            data_path = os.path.join(os.getenv("CHUNK_PATH"), strategy)

            # Supprimer la table si elle existe déjà
            if db.table_names() and table_name in db.table_names():
                db.drop_table(table_name)
                
            # Créer la table
            table = db.create_table(
                name=f"chunks_{model}_{strategy}",
                schema=Chunk
            )

            
            for file in os.listdir(data_path):
                with open(os.path.join(data_path, file), 'r') as f:
                    data = json.load(f)
                table.add(data)

if __name__ == "__main__":
    os.chdir("/home/maxime/gesida-rag-huil/")
    create_collection(["jina"], ["nat", "nat_sem"])