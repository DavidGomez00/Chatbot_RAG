import json
import os
import time
import logging
import argparse
import lancedb

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


def initialize_collection(database_path:str, strategy:str, embedding_model:str):
    # [ Initialize collection ]
    db_path = os.path.join(os.path.join(database_path, embedding_model), strategy)
    return lancedb.connect(db_path)


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
    embedding_model = args.model
    llm = args.llm

    LANCE_DB_PATH, N_RESULTS, SEMANTIC_MODEL, DATA_PATH, OLLAMA_SERVER_IP, RESULT_PATH = load_enviroment()

    # Crear directorio para guardar resultados
    os.makedirs(os.path.join(RESULT_PATH, llm, embedding_model, chunk_str), exist_ok=True)

    # [ Initialize collection ]
    collection = initialize_collection(
        database_path=LANCE_DB_PATH,
        strategy=chunk_str,
        embedding_model=embedding_model
    )

    # [ Process queries ]
    start_time = time.time()
    logger.info("Processing queries...")

    base_prompt = read_file("system_prompt.txt")
    queries = read_file("queries.txt", split_lines=True)

    ollama = OllamaLLM(model=llm, base_url=OLLAMA_SERVER_IP)

    for query_id, query in enumerate(queries):
        try: 
            # Get context and answer
            tbl = collection.open_table("chunks")
            # Get list of 3 most relevant in table
            results = tbl.search(query).limit(N_RESULTS).to_list()

            print(results[0]) 
            break

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
