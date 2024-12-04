import json
import os
import time
import logging
import argparse

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from semantic_db import create_collection
from setup import setup

load_dotenv()

def load_enviroment():
    # [ Read environment variables ]
    try:
        CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE"))
        N_RESULTS = int(os.getenv("N_RESULTS"))
        SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL")
        DATA_PATH = os.getenv("DATA_PATH")
        OLLAMA_SERVER_IP = os.getenv("OLLAMA_SERVER_IP")
        if not all([CHUNK_MAX_SIZE, N_RESULTS, SEMANTIC_MODEL, DATA_PATH, OLLAMA_SERVER_IP]):
            raise ValueError("Missing or invalid environment variables.")
    except Exception as e:
        print(f"Error with environment variables: {e}")
        exit(1)
    return CHUNK_MAX_SIZE, N_RESULTS, SEMANTIC_MODEL, DATA_PATH, OLLAMA_SERVER_IP
    

def create_directories(DATA_PATH, llm, model, chunk_str):
    # [ Create directories ]
    os.makedirs(os.path.join(DATA_PATH, "DocsVIH"), exist_ok=True)
    result_path = os.path.join(DATA_PATH, "results", llm, model, chunk_str)
    os.makedirs(result_path, exist_ok=True)
    return result_path


def initialize_collection(chunk_str, CHUNK_MAX_SIZE, SEMANTIC_MODEL):
    # [ Initialize collection ]
    setup(strategy=chunk_str,
          chunk_size=CHUNK_MAX_SIZE,
          semantic_model=SEMANTIC_MODEL)
    return create_collection(model=model, strategy=chunk_str)


def read_file(file, split_lines=False):
    # [ Read file from DATA_PATH ]
    file_path = os.path.join(DATA_PATH, file)
    if not os.path.isfile(file_path):
        logger.error(f"Missing {file} in {DATA_PATH}.")
        exit(1)
    with open(file_path, "r") as f:
        return f.readlines() if split_lines else f.read()


def parse_arguments():
    # [ Parse arguments ]
    parser = argparse.ArgumentParser(description="Process semantic queries.")
    parser.add_argument("--chunk_str", required=True, help="Chunking strategy (e.g., 'nat' or 'nat_sem').")
    parser.add_argument("--model", required=True, help="Semantic model to use (e.g., 'BGE-M3').")
    parser.add_argument("--llm", required=True, help="Language model to use (e.g., 'nemotron').")
    return parser.parse_args()


if __name__ == "__main__":

    # [ Set up logging ]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # [ Parse arguments ]
    args = parse_arguments()
    chunk_str = args.chunk_str
    model = args.model
    llm = args.llm

    CHUNK_MAX_SIZE, N_RESULTS, SEMANTIC_MODEL, DATA_PATH, OLLAMA_SERVER_IP = load_enviroment()

    # Crear directorio para guardar resultados
    RESULT_PATH = create_directories(DATA_PATH, llm, model, chunk_str)

    # [ Generate chunks and create collection ]
    logger.info("Starting setup...")
    collection = initialize_collection(chunk_str, CHUNK_MAX_SIZE, SEMANTIC_MODEL)
    logger.info("Done")

    # [ Process queries ]
    start_time = time.time()
    logger.info("Processing queries...")

    base_prompt = read_file("system_prompt.txt")
    queries = read_file("queries.txt", split_lines=True)

    ollama = OllamaLLM(model=llm, base_url=OLLAMA_SERVER_IP)

    for query_id, query in enumerate(queries):
        try: 
            # Get context and answer
            results = collection.query(query_texts=[query], n_results=N_RESULTS)
            documents = results['documents'][0][:N_RESULTS]
            metadatas = results['metadatas'][0][:N_RESULTS]
            context = "\n".join(documents)
            context_list = [{"document_id": meta["document_id"], "section_id": meta["section_id"]} for meta in metadatas]
            messages = [
                {"role": "system", "content": f"{base_prompt}\n{{ {context} }}"},
                {"role": "user", "content": query.strip()}
            ]
            answer = ollama.invoke(messages)
            
            # Save query infos
            query_info = {
                "query": query,
                "context": context_list,
                "context_len": len(context),
                "answer": answer
            }
            output_file = os.path.join(RESULT_PATH, f"query_{query_id}.json")
            with open(output_file, 'w') as f:
                json.dump(query_info, f, indent=4)

            logger.info(f"Query {query_id} processed successfully.")

        except Exception as e:
            logger.error(f"Error processing query {query_id}: {e}")
            continue

    logger.info(f"--- Execution finished in {(time.time() - start_time):.2f} seconds ---")
