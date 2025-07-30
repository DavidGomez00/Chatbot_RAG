import os
import sys
import json
import logging
import requests
import argparse
import pandas as pd
import lancedb
from lancedb.embeddings import EmbeddingFunctionRegistry
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath('/home/maxime/gesida-rag-huil/'))

from Chunk_preparation.database import RemoteOllamaEmbeddings
from langchain_ollama import OllamaLLM

load_dotenv()

def load_environment():
    try:
        N_RESULTS = int(os.getenv("N_RESULTS"))
        DATA_PATH = os.getenv("DATA_PATH")
        LANCE_DB_PATH = os.getenv("LANCE_DB_PATH")
        SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL")
        OLLAMA_SERVER_IP = os.getenv("OLLAMA_SERVER_IP")
        CSV_PATH = os.getenv("CSV_PATH")
        NEW_RESULT_PATH = os.getenv("NEW_RESULT_PATH")
        if not all([LANCE_DB_PATH, N_RESULTS, SEMANTIC_MODEL, DATA_PATH, OLLAMA_SERVER_IP, CSV_PATH, NEW_RESULT_PATH]):
            raise ValueError("Missing or invalid environment variables.")
    except Exception as e:
        raise ValueError(f"Error reading environment variables: {e}")
    return LANCE_DB_PATH, N_RESULTS, SEMANTIC_MODEL, DATA_PATH, OLLAMA_SERVER_IP, CSV_PATH, NEW_RESULT_PATH

def read_file(file, split_lines=False):
    with open(file, "r") as f:
        return f.readlines() if split_lines else f.read()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process specific semantic queries.")
    parser.add_argument("--llm", required=True, help="Language model to use (e.g., 'nemotron').")
    return parser.parse_args()

def read_queries_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df['Question Text'].tolist()

def read_comments_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df['Comments'].tolist()

def read_answers_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df['Answers'].tolist()

def analyze_comments(comments):
    issues = []
    for comment in comments:
        if not comment:
            issues.append("No comment")
            continue
        comment = comment.lower()
        if "largo" in comment:
            issues.append("Response too long")
        elif "partiel" in comment:
            issues.append("Partial response")
        elif "hors sujet" in comment:
            issues.append("Off-topic response")
        else:
            issues.append("Unknown issue")
    return issues

def process_query(query_id, query, tbl, base_prompt, llm, embedding_model, strategy, comments, old_answers):
    logger = logging.getLogger()
    logger.info(f"Starting processing of Query {query_id + 1}")

    results = tbl.search(query).limit(5).to_list()
    logger.info(f"Retrieved results for Query {query_id + 1}")

    context_info = [
        {
            "text": result["text"],
            "source": f"{result['document_id']}, section {result['section_id']}",
            "_distance": result["_distance"]
        } for result in results
    ]
    context = "\n".join(result["text"] for result in results)

    issues = analyze_comments([comments[query_id]])
    issue = issues[0]

    messages = [
        {"role": "system", "content": f"Eres un asistente especializado en responder consultas sobre el VIH, con el siguiente conocimiento.\n{context}\n{base_prompt}\nProblème avec l'ancienne réponse: {issue}\nAncienne réponse: {old_answers[query_id]}"},
        {"role": "user", "content": query.strip()}
    ]

    logger.info(f"Sending request for Query {query_id + 1}")
    answer = requests.post(
        url=f"http://{os.getenv('OLLAMA_SERVER_IP')}/api/chat",
        json={
            "model": llm,
            "messages": messages,
            "stream": False
        }
    ).json()["message"]["content"].replace("*", "")

    logger.info(f"Received response for Query {query_id + 1}")
    query_info = {
        "query": query,
        "context_info": context_info,
        "answer": answer
    }

    output_path = os.path.join(os.getenv("NEW_RESULT_PATH"), llm, embedding_model, strategy)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"query_{query_id}.json")

    with open(output_file, 'w') as f:
        json.dump(query_info, f, indent=4)

    logger.info(f"Query {query_id + 1} processed successfully.")
    return f"Query {query_id + 1} processed successfully."

if __name__ == "__main__":
    os.chdir("/home/maxime/gesida-rag-huil/")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    args = parse_arguments()
    llm = args.llm

    LANCE_DB_PATH, N_RESULTS, SEMANTIC_MODEL, DATA_PATH, OLLAMA_SERVER_IP, CSV_PATH, NEW_RESULT_PATH = load_environment()

    queries = read_queries_from_csv(CSV_PATH)
    old_answers = read_answers_from_csv(CSV_PATH)
    comments = read_comments_from_csv(CSV_PATH)

    embedding_models = ["BGE-M3", "jina"]
    strategies = ["nat", "nat-sem"]

    for embedding_model in embedding_models:
        for strategy in strategies:
            logger.info(f"--- Processing {embedding_model} with {strategy} ---")
            db = lancedb.connect(LANCE_DB_PATH)

            # Ouvrir la table avec le modèle d'embedding spécifié
            tbl = db.open_table(f"chunks_{embedding_model}_{strategy}")

            base_prompt = read_file("system_prompt.txt")

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for query_id, query in enumerate(queries):
                    futures.append(executor.submit(process_query, query_id, query, tbl, base_prompt, llm, embedding_model, strategy, comments, old_answers))

                for future in futures:
                    future.result()

    logger.info(f"--- Execution finished ---")
