import json
import os
import re
import warnings

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from semantic_db import embedding_model_dict


load_dotenv()


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


def _fixed_size_chunking(document:str, text:list, chunk_size:int=512) -> list:
    '''Divides each of the provided texts into chunks of a fixed size. The chunks are
       created by splitting the text by sentences, avoiding cutting sentences in the middle.'''
    chunks = []
    for section in text:
        chunk_id = 0
        # Get the section id
        section_id = re.search(r'((?:\d+\.){2,}\s)', section).group(0)[:-1]
        # Remove the section index from the section text
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
                # Add the current chunk to the chunks
                chunks.append({"chunk_id": document+'_'+section_id+'_'+str(chunk_id),
                            "section_id": section_id,
                            "document_id":document,
                            "text": current_chunk})
                chunk_id += 1
                current_chunk = ''
            # Add the sentence to the current chunk
            current_chunk += sentence + ' '
        if (not current_chunk.isspace()) and (current_chunk != ''):
            # Add the last chunk
            chunks.append({"chunk_id": document+'_'+section_id+'_'+str(chunk_id),
                        "section_id": section_id,
                        "document_id":document,
                        "text": current_chunk})
    return chunks


def natural_section_split(document:str, text:str, chunk_size:int=512) -> list:
    '''Split the text into natural sections of the text. Then, split sections bigger
       than a fixed size into two or more chunks, avoiding cutting sentences in the
       middle.'''
    # Get sections
    sections = _get_sections(text)
    # Divide sections into chunks
    chunks = _fixed_size_chunking(document=document,
                                 text=sections,
                                 chunk_size=chunk_size)
    return chunks


def semantic_split(text:str, semantic_model="multilingual5e"):
    '''Divide a given text into chunks using semantic splitting'''
    hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[semantic_model])
    text_splitter = SemanticChunker(hf_embeddings,
                                    buffer_size=1,
                                    breakpoint_threshold_type='percentile',
                                    breakpoint_threshold_amount=95)
    return text_splitter.create_documents([text])


def natural_and_semantic_split(document:str, text:str, semantic_model="multilingual5e"):
    '''Split the text recursively'''
    sections = _get_sections(text)
    chunk_list = []
    
    for section in sections:
        chunk_id = 0
        # Get section identifier
        section_id = re.search(r'((?:\d+\.){2,}\s)', section).group(0)[:-1]
        # Remove the section index from the section text
        section = section[len(section_id):]
        # Split the section into chunks
        chunks = semantic_split(section, semantic_model)
        for chunk in chunks:
            if chunk.page_content.isspace() or chunk.page_content == '':
                continue
            c_dict = {"chunk_id": document+'_'+section_id+'_'+str(chunk_id),
                      'document_id': document,
                      'section_id': section_id,
                      'text': chunk.page_content}
            chunk_list.append(c_dict)
            chunk_id += 1

    return chunk_list


def create_chunks(documents:list, strategy:str, chunk_size:int=512, semantic_model:str="multilingual5e") -> list:
    '''Create chunks from the parsed documents using the selected strategy'''
    for document in documents:
        if not os.path.exists('.data/parsed_documents/'+document):
            warnings.warn(f"El documento {document} no se encuentra entre los documentos procesados \
                           y será ignorado. Por favor, asegúrese de que se ha procesado correctamente \
                          o que el nombre del documento sea correcto.")
            continue

        text = open('.data/parsed_documents/'+document, 'r').read()
        if strategy == "nat":
            chunks = natural_section_split(document=document.split(".")[0],
                                           text=text,
                                           chunk_size=chunk_size)
        elif strategy == "nat_sem":
            chunks = natural_and_semantic_split(document=document.split(".")[0],
                                                text=text,
                                                semantic_model=semantic_model)
        else:
            raise NotValidStrategy(f"La estrategia {strategy} no es válida. Las opciones son nat|nat_sem")
        
        # Save the chunks into a file
        chunk_path = os.path.join(os.getenv("DATA_PATH"), "chunks/")
        if not os.path.exists(chunk_path):
            os.makedirs(chunk_path)
        saving_path = os.path.join(chunk_path, strategy)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        with open(os.path.join(saving_path, document.split(".")[0]+'.json'), 'w') as f:
            f.write(json.dumps(chunks))


if __name__ == "__main__":
    text = open('.data/parsed_documents/ManualClinicoVIH.txt', 'r').read()
    chunks = natural_section_split("test", text)
    for c in chunks:
        print(c["chunk_id"])