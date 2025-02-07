import json
import os

import logging
import requests
import argparse
import lancedb
from lancedb.embeddings import EmbeddingFunctionRegistry
from Chunk_preparation.database import RemoteOllamaEmbeddings

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()

def load_enviroment():
    # [ Read environment variables ]
    try:
        N_RESULTS = int(os.getenv("N_RESULTS"))
        DATA_PATH = os.getenv("DATA_PATH")
        RESULT_PATH = os.getenv("RESULT_PATH")
        LANCE_DB_PATH = os.getenv("LANCE_DB_PATH")
        SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL")
        OLLAMA_SERVER_IP = os.getenv("OLLAMA_SERVER_IP")
        if not all([LANCE_DB_PATH, N_RESULTS, SEMANTIC_MODEL, DATA_PATH, OLLAMA_SERVER_IP]):
            raise ValueError("Missing or invalid environment variables.")
        
    except Exception as e:
        raise ValueError(f"Error reading environment variables: {e}")
    
    return LANCE_DB_PATH, N_RESULTS, SEMANTIC_MODEL, DATA_PATH, OLLAMA_SERVER_IP, RESULT_PATH


def read_file(file, split_lines=False):
    with open(file, "r") as f:
        return f.readlines() if split_lines else f.read()


def parse_arguments():
    # [ Parse arguments ]
    parser = argparse.ArgumentParser(description="Process semantic queries.")
    parser.add_argument("--llm", required=True, help="Language model to use (e.g., 'nemotron').")
    return parser.parse_args()


if __name__ == "__main__":

    os.chdir("/home/david/GitHub/gesida-rag-huil/")

    # [ Set up logging ]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # [ Parse arguments ]
    args = parse_arguments()
    llm = args.llm

    embedding_models = ["BGE-M3", "jina"]
    strategies = ["nat", "nat-sem"]

    for embedding_model in embedding_models:
        for strategy in strategies:
            logger.info(f"--- Processing {embedding_model} with {strategy} ---")

            # [ Get database ]
            collection = lancedb.connect(os.getenv("LANCE_DB_PATH"))
            registry = EmbeddingFunctionRegistry.get_instance()

            # [ Connect to Ollama service]
            # ollama = OllamaLLM(model=llm, base_url=os.getenv("OLLAMA_SERVER_IP"))

            # [ Process queries ]
            base_prompt = read_file("system_prompt.txt")
            queries = read_file("queries.txt", split_lines=True)
            for query_id, query in enumerate(queries):

                logger.info(f"Query {query_id + 1} of {len(queries)}")
                
                tbl = collection.open_table(f"chunks_{embedding_model}_{strategy}")
                results = tbl.search(query).limit(5).to_list()
                context_info =[
                    {
                        "text": result["text"],
                        "source": f"{result['document_id']}, section {result['section_id']}",
                        "_distance": result["_distance"]
                    } for result in results
                ]
                context = "\n".join(result["text"] for result in results)
                messages = [
                    {"role": "user", "content": f"Eres un asistente especializado en responder consultas sobre el VIH, con el siguiente conocimiento.\n{context}\n{base_prompt}\n{query.strip()}"},
                ]
                # answer = ollama.invoke(messages).replace("*", "")
                answer = requests.post(
                    url=f"http://{os.getenv('OLLAMA_SERVER_IP')}/api/chat",
                    json={
                        "model": llm,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.6}
                    }
                ).json()["message"]["content"].replace("*", "")

                # Save query info
                query_info = {
                    "query": query,
                    "context_info": context_info,
                    "answer": answer
                }
                
                output_path = os.path.join(os.getenv("RESULT_PATH"), llm, embedding_model, strategy)
                os.makedirs(output_path, exist_ok=True)
                output_file = os.path.join(output_path, f"query_{query_id}.json")
                with open(output_file, 'w') as f:
                    json.dump(query_info, f, indent=4)

                logger.info(f"Query {query_id + 1} processed successfully.")

    logger.info(f"--- Execution finished ---")
