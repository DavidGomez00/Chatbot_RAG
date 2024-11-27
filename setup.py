import os

from dotenv import load_dotenv

from chunker import create_chunks
from files_parser import parse_files


def setup(strategy:str="nat", chunk_size:int=512, semantic_model:str="multilingual5e", create_new:bool=False):
    '''Process the data and divide it into chunks. Save the chunks in the data folder'''
    load_dotenv()
    if not os.path.exists(os.getenv("DATA_PATH")):
        os.makedirs(os.getenv("DATA_PATH"))
    DOCS_PATH = os.path.join(os.getenv("DATA_PATH"), 'DocsVIH/')
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
    PARSED_PATH = os.path.join(os.getenv("DATA_PATH"), 'parsed_documents/')
    if not os.path.exists(PARSED_PATH):
        os.makedirs(PARSED_PATH)

    files = os.listdir(DOCS_PATH)
    to_parse = []
    for file in files:
        if not os.path.exists(os.path.join(PARSED_PATH, file)):
            to_parse.append(file)

    parsed_files = parse_files(to_parse)
        
    # Divide the text in the txt files into chunks
    create_chunks(documents=parsed_files,
                  strategy=strategy,
                  chunk_size=chunk_size,
                  semantic_model=semantic_model
                  )


if __name__ == "__main__":

    setup(override=False)

