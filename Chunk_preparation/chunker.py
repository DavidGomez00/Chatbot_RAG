import json
import os
import re
import torch
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
PARSED_PATH, CHUNK_PATH = os.getenv("PARSED_PATH"), os.getenv("CHUNK_PATH")

embedding_model_dict = {
    "multilingual5e": 'intfloat/multilingual-e5-large',
    "BGE-M3": "BAAI/bge-m3"
}


class NotValidStrategy(ValueError):
    '''Exception when the strategy introduced is not valid'''
    def __init__(self, message, *args):         
        super(NotValidStrategy, self).__init__(message, *args)


def _get_sections(text:str) -> list:
    '''Divide the text into sections using a regex pattern'''
    # Use regex to find the patterns like "X.X. <text>" without <text>
    pattern = re.compile(r'((?:\d+\.){2,}\s)')
    # Split the text by the pattern and keep the delimiters
    parts = pattern.split(text)
    # Combine the delimiters with the following text
    sections = ["".join(x) for x in zip(parts[1::2], parts[2::2])]
    return sections


def _fixed_size_chunking(document:str, sections:list, chunk_size:int=512) -> list:
    '''Divides each of the provided texts into chunks of a fixed size. The chunks are
       created by splitting the text by sentences, avoiding cutting sentences in the middle.'''
    
    chunks = []
    for section in sections:
        chunk_id = 0
        # Get the section id
        section_id = re.search(r'((?:\d+\.){2,}\s)', section).group(0)[:-1]
        # Remove the index from the section text
        section = section[len(section_id):]
        # Split the section into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', section)
        # Initialize the current chunk
        current_chunk = ''
        
        for sentence in sentences:
            # If the sentence is empty or made of only spaces, skip it
            if sentence.isspace() or sentence == '':
                continue

            # If the current chunk plus the sentence is larger than the chunk size
            if len(current_chunk) + len(sentence) > chunk_size:
                chunk = {
                    "chunk_id": document+'_'+section_id+'_'+str(chunk_id),
                    "section_id": section_id,
                    "document_id":document,
                    "text": current_chunk
                }
                # Add the current chunk to the chunks
                if chunk['text']:
                    chunks.append(chunk)
                    chunk_id += 1
                    current_chunk = ''
            
            # Add the sentence to the current chunk
            current_chunk += sentence + ' '

        # Add the last chunk
        if (not current_chunk.isspace()) and current_chunk:
            chunk = {
                "chunk_id": document+'_'+section_id+'_'+str(chunk_id),
                "section_id": section_id,
                "document_id":document,
                "text": current_chunk
            }
            if chunk['text']:
                chunks.append(chunk)

    return chunks


def _natural_section_split(document:str, text:str, chunk_size:int=512) -> list:
    '''Split the text into natural sections of the text. Then, split sections bigger
       than fixed size into two or more chunks, avoiding cutting sentences in the
       middle.'''
    
    # Divide sections into chunks
    chunks = _fixed_size_chunking(
        document=document,
        sections=_get_sections(text),
        chunk_size=chunk_size
    )
    return chunks

# Désactive l'utilisation du GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Libérer la mémoire GPU
torch.cuda.empty_cache()
def _semantic_split(text: str, semantic_model="multilingual5e"):
    '''Divide a given text into chunks using semantic splitting.'''

    text_splitter = SemanticChunker(
        embeddings=HuggingFaceEmbeddings(
            model_name=embedding_model_dict[semantic_model],
            model_kwargs={'device': 'cpu'}  # Assurez-vous que le modèle utilise le CPU
        ),
        buffer_size=1,
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=95
    )
    return text_splitter.create_documents([text])



def _natural_and_semantic_split(document:str, text:str, semantic_model="multilingual5e"):
    '''Split the text into natural sections, then split each section
    using semantic splitting.'''

    chunk_list = []
    for section in _get_sections(text):
        chunk_id = 0
        section_id = re.search(r'((?:\d+\.){2,}\s)', section).group(0)[:-1]
        # Remove the index from the section text
        section = section[len(section_id):]

        # Split the section into chunks
        chunks = _semantic_split(section, semantic_model)

        # Add chunks to final list
        for chunk in chunks:
            if chunk.page_content.isspace() or not chunk.page_content:
                continue
            c_dict = {
                "chunk_id": document+'_'+section_id+'_'+str(chunk_id),
                'document_id': document,
                'section_id': section_id,
                'text': chunk.page_content
            }
            chunk_list.append(c_dict)
            chunk_id += 1

    return chunk_list


def _create_chunks_from_file(document:str, strategy:str, chunk_size:int=512, semantic_model:str="multilingual5e") -> list:
    '''Create chunks from given file using the given strategy'''

    text = open(os.path.join(PARSED_PATH, document), 'r').read()

    if strategy == "nat":
        chunks = _natural_section_split(
            document=document.split(".")[0],
            text=text,
            chunk_size=chunk_size
        )
    elif strategy == "nat_sem":
        chunks = _natural_and_semantic_split(
            document=document.removesuffix('.txt'),
            text=text,
            semantic_model=semantic_model
        )
    else:
        raise NotValidStrategy(
            f"La estrategia {strategy} no es válida. Las opciones son nat|nat_sem."
        )
    
    # Save the chunks into a json file
    saving_path = os.path.join(CHUNK_PATH, strategy)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    with open(os.path.join(saving_path, document[:-4]+'.json'), 'w') as f:
        json.dump(chunks, f, indent=4)


def create_chunks(strategy:str, chunk_size:int=512, semantic_model:str="multilingual5e") -> list:
    '''Create chunks from PARSED_PATH files using the given strategy'''

    for document in os.listdir(PARSED_PATH):
        text = open(os.path.join(PARSED_PATH, document), 'r').read()

        if strategy == "nat":
            chunks = _natural_section_split(
                document=document.split(".")[0],
                text=text,
                chunk_size=chunk_size
            )
        elif strategy == "nat_sem":
            chunks = _natural_and_semantic_split(
                document=document.removesuffix('.txt'),
                text=text,
                semantic_model=semantic_model
            )
        else:
            raise NotValidStrategy(
                f"La estrategia {strategy} no es válida. Las opciones son nat|nat_sem."
            )
        
        # Save the chunks into a json file
        saving_path = os.path.join(CHUNK_PATH, strategy)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        with open(os.path.join(saving_path, document[:-4]+'.json'), 'w') as f:
            json.dump(chunks, f, indent=4)


if __name__ == "__main__":
    _create_chunks_from_file("ManualClinico.txt", "nat_sem")
    _create_chunks_from_file("ManualClinico.txt", "nat")