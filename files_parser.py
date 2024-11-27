import os
import re
import warnings

from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

# Ensure the data path exists
DATA_PATH = os.getenv("DATA_PATH")
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
DOCS_PATH = os.path.join(DATA_PATH, 'DocsVIH')
if not os.path.exists(DOCS_PATH):
    os.makedirs(DOCS_PATH)
PARSED_PATH = os.path.join(DATA_PATH, 'parsed_documents') 
if not os.path.exists(os.path.join(PARSED_PATH)):
    os.makedirs(PARSED_PATH)



def normalize_text(text):
    '''Normalize the text by removing extra spaces and new lines'''
    
    # Sustituir saltos de línea en mitad de una frase por un espacio
    text = re.sub(r'(?<![.!?])\n', ' ', text)
    # Normalizar espacios en blanco. Cualquier cantidad de espacios en blanco unificados a un solo espacio
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text

def parse_files(files:list) -> list:
    '''Parse the files in the list and save the text to a new file'''
    parsed_files = []
    for document in files:
        if not os.path.exists(os.path.join(DOCS_PATH, document)):
            warnings.warn(f"El documento {document} no se encuentra entre los documentos y no será \
                          procesado. Por favor, asegúrate de que el documento existe.")
            continue

        if document.endswith('.pdf'):
            reader = PdfReader(os.path.join(DOCS_PATH, document))
            texts = [page.extract_text() for page in reader.pages]
        else:
            with open(os.path.join(DOCS_PATH, document), "r") as f:
                texts = f.readlines()
            texts = [text for text in texts if text]
        text = "\n".join(texts)
        text = normalize_text(text)

        if document.endswith('.pdf'):
            document = document.replace('.pdf', '.txt')
        file_name_out = os.path.join(PARSED_PATH, document)

        with open(file_name_out, "w") as f:
            f.write(text)
        parsed_files.append(document)
    
    return parsed_files


if __name__ == "__main__":

    parse_files(["manejo-compartido-del-paciente-con-infeccion-por-vih.txt"])
        
