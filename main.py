import json
import os
import time

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from semantic_db import create_collection
from setup import setup

load_dotenv()


# Global Variables
CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE"))
N_RESULTS = int(os.getenv("N_RESULTS"))
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL")
DATA_PATH = os.getenv("DATA_PATH")

# Check if the data path exists
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(os.path.join(DATA_PATH, "DocsVIH")):
    os.makedirs(os.path.join(DATA_PATH, "DocsVIH"))
PDF_PATH = os.path.join(DATA_PATH, "DocsVIH")


if __name__ == "__main__":

    # Select parameters
    #chunk_str = "nat_sem"
    chunk_str = "nat"

    #model = 'multilinguale5'
    model = "BGE-M3"

    #llm = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    #llm = "mistralai/Mistral-7B-Instruct-v0.1"
    #llm = "BioMistral/BioMistral-7B-DARE"
    llm = "nemotron"

    # Crear directorio para guardar resultados
    RESULT_PATH = os.path.join(DATA_PATH, "results", llm, model, chunk_str)
    os.makedirs(RESULT_PATH, exist_ok=True)
    
    # Generar los chunks
    print("Setting up...")
    setup(strategy=chunk_str,
          chunk_size=CHUNK_MAX_SIZE,
          semantic_model=SEMANTIC_MODEL
          )
    print("Done")
    print("Initializing vector database...")
    # Obtener la base de datos vectorial
    collection = create_collection(model=model,
                                   strategy=chunk_str
                                   )
    print("Done")

    # Read the system prompt
    with open(os.path.join(DATA_PATH, "system_prompt.txt"), "r") as f:
            base_prompt = f.read()
    
    # Read all queries from the queries.txt file
    with open(os.path.join(DATA_PATH, "queries.txt"), 'r') as f:
        queries = f.readlines()

    start_time = time.time()

    # Inicializar el modelo
    ollama = OllamaLLM(model=llm, base_url='176.98.223.168')

    # Procesar las consultas
    for query_id, query in enumerate(queries):
        # Búsqueda semántica de documentos relevantes
        results = collection.query(query_texts=[query], n_results=N_RESULTS)
        data = {
            'Document ID': [results['metadatas'][0][i]['document_id'] for i in range(N_RESULTS)],
            'Section ID': [results['metadatas'][0][i]['section_id'] for i in range(N_RESULTS)],
            'Text': [results['documents'][0][i] for i in range(N_RESULTS)]
        }
        context = "\n".join(data['Text'])
        messages = [
            {"role": "system", "content": base_prompt + "\n{"+context+"}"},
            {"role": "user", "content": query}
            ]
        
        # Generar respuesta
        answer = ollama.invoke(messages)
        
        # Guardar resultados
        with open(os.path.join(RESULT_PATH, f"query_{query_id}.json"), 'w') as f:
            f.write(json.dumps(answer, indent=4))
        
        print(f"Query {query_id} done")


    print("Done")
    print(f"--- La ejecución ha tardado {(time.time() - start_time):.2f} segundos ---")
