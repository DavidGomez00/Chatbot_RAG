## dans ce script on va nettoyer le texte extrait, et sauvegarder le texte traité dans un autre répertoire.

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


def _clean_text(text):
    '''Normalize the text by removing extra spaces and new lines'''

    # Replace new lines with spaces
    text = re.sub(r'(?<![.!?])\n', ' ', text)
    
    # Remove extra blank spaces.
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text

def _parse_file(document):
     # Extract text from pdf, txt files
    if document.endswith('.pdf'):
        reader = PdfReader(os.path.join(DOCS_PATH, document))
        texts = [page.extract_text() for page in reader.pages]

    elif document.endswith('.txt'):
        with open(os.path.join(DOCS_PATH, document), "r") as f:
            texts = f.readlines()
        texts = [text for text in texts if text]
    
    else:
        warnings.warn(f"Document {document} is not a .pdf or .txt file. Skipping...")

    # Clean text
    text = "\n".join(texts)
    text = _clean_text(text)

    # Save document in PARSED_PATH
    file_name_out = os.path.join(PARSED_PATH, document[:-3]+"txt")
    with open(file_name_out, "w") as f:
        f.write(text)


def parse_files():
    '''Read documents from DOCS_PATH and save them in PARSED_PATH'''
    
    for document in os.listdir(DOCS_PATH):
        
        # Extract text from pdf, txt files
        if document.endswith('.pdf'):
            reader = PdfReader(os.path.join(DOCS_PATH, document))
            texts = [page.extract_text() for page in reader.pages]

        elif document.endswith('.txt'):
            with open(os.path.join(DOCS_PATH, document), "r") as f:
                texts = f.readlines()
            texts = [text for text in texts if text]
        
        else:
            warnings.warn(f"Document {document} is not a .pdf or .txt file. Skipping...")
            continue

        # Clean text
        text = "\n".join(texts)
        text = _clean_text(text)

        # Save document in PARSED_PATH
        file_name_out = os.path.join(PARSED_PATH, document[:-3]+"txt")
        with open(file_name_out, "w") as f:
            f.write(text)