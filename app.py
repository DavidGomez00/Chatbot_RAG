import os

import streamlit as st
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

import time

from semantic_db import create_collection
from setup import setup


# Global Variables
CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE"))
N_RESULTS = int(os.getenv("N_RESULTS"))
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL")
if not os.path.exists(os.getenv("DATA_PATH")):
    os.makedirs(os.getenv("DATA_PATH"))
if not os.path.exists(os.path.join(os.getenv("DATA_PATH"), "DocsVIH")):
    os.makedirs(os.path.join(os.getenv("DATA_PATH"), "DocsVIH"))
PDF_PATH = os.path.join(os.getenv("DATA_PATH"), "DocsVIH")

model_dict = { "Modelo 1 - multilinguale5": 'multilinguale5',
               "Modelo 2 - paraphrase": "paraphrase",
               "Modelo 3 - BGE-M3": "BGE-M3",
               "Modelo 4 - RoBERTa-base-biomedical-clinical-es": "RoBERTa-base-biomedical-clinical-es"}

llm_dict = {"LLM 1 - Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "LLM 2 - Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.1"
            }

chunk_dict = { "Natural chunk": "nat", "Semantic": "nat_sem"}


pdf_files = [os.path.join(PDF_PATH, f) for f in os.listdir(PDF_PATH) if f.endswith('.pdf')]


if __name__ == "__main__":

    # Set title
    st.write("# Evaluación de modelos de embedding para RAG")
    
    # Check if data needs to be overriden
    st.session_state.create_new = False
    if st.checkbox("¿Se han actualizado los PDFs?"):
        st.session_state.create_new = True

    # Select parameters
    st.write("## Selección de parámetros")
    chunk = st.selectbox(
            label='Método de divisón en *chunks*',
            options=chunk_dict.keys(),
        )

    model = st.selectbox(
            label='Escoja un modelo para ser evaluado',
            options=model_dict.keys(),
        )

    llm = st.selectbox(
            label='Escoja un LLM para ser evaluado',
            options=llm_dict.keys(),
        )
    
    # query = st.text_area(label="Consulta", value="¿Qué es el VIH? ¿Cuáles son sus síntomas?")

    if st.button('Realizar consulta'):
        # Generar los chunks
        print("Setting up...")
        st.session_state.chunk_str = chunk_dict[chunk]
        setup(strategy=chunk_dict[chunk],
              chunk_size=CHUNK_MAX_SIZE,
              semantic_model=SEMANTIC_MODEL
              )
        print("Done")
        print("Initializing vector database...")
        # Obtener la base de datos vectorial
        model_name = model_dict[model]
        st.session_state.collection = create_collection(model=model_name,
                                                        strategy=st.session_state.chunk_str
                                                        )
        print("Done")

        # Read the system prompt
        with open(".data/system_prompt.txt", "r") as f:
                base_prompt = f.read()
        
        # Read all queries from the queries.txt file
        with open(".data/queries.txt", 'r') as f:
            queries = f.readlines()

        start_time = time.time()

        # Crear directorio para guardar resultados
        DATA_PATH = os.getenv("DATA_PATH")
        RESULT_PATH = os.path.join(DATA_PATH, "results/")
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        RESULT_PATH = os.path.join(RESULT_PATH, llm_dict[llm])
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        RESULT_PATH = os.path.join(RESULT_PATH, model_dict[model])
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        RESULT_PATH = os.path.join(RESULT_PATH, chunk_dict[chunk])
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)

        # Procesar las consultas
        for query_id, query in enumerate(queries):
            # Búsqueda semántica de documentos relevantes
            results = st.session_state.collection.query(query_texts=[query], n_results=N_RESULTS)
            data = {
                'Document ID': [results['metadatas'][0][i]['document_id'] for i in range(N_RESULTS)],
                'Section ID': [results['metadatas'][0][i]['section_id'] for i in range(N_RESULTS)],
                'Text': [results['documents'][0][i] for i in range(N_RESULTS)]
            }
            context = "\n".join(data['Text'])
            messages = [
                {"role": "system", "content": base_prompt + "{"+context+"}"},
                {"role": "user", "content": query}
                ]
            if llm_dict[llm] == "meta-llama/Meta-Llama-3.1-8B-Instruct":
                pipeline = transformers.pipeline('text-generation',
                                                model=llm_dict[llm],
                                                model_kwargs={"torch_dtype": torch.bfloat16},
                                                device_map="auto"
                                                )
                answer = pipeline(messages, max_new_tokens=1000)
                st.write(answer[0]['generated_text'])
            else:
                model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
                                                            torch_dtype=torch.float16,
                                                            device_map="auto")
                tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
                input_ids = tokenizer.apply_chat_template(messages,
                                                        tokenize=True,
                                                        return_tensors="pt",
                                                        add_generation_prompt=True)
                with torch.no_grad():
                    generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
                decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                st.write(decoded[0])
                # Save result:
                with open(os.path.join(RESULT_PATH, f"{query_id}.txt"), "w") as f:
                    f.write(decoded[0])
        print("Done")
        print(f"--- La ejecución ha tardado {(time.time() - start_time):.2f} segundos ---")
