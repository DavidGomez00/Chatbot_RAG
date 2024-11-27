# ChatBotRAG

## Data

[UPM DRIVE](https://drive.upm.es/s/OnAXLBbk8mQdvwR)

## Env
should be define in .env 
variables: 
- PATH_DATA : the path that saves all the pdfs

## Estructura 
La app estara divida en modulos:
- [Modulo de carga de archivos](./files_parser.py): Dado un path donde está guardado el documento, lee el pdf y limpia el contenido, obteniendo un archivo de texto plano con toda la info relevante del documento. El script debe ejecutarse pasando como argumento los nombres de los documentos en pdf. El script busca los documentos con ese mismo nombre en el directorio .data/DocsVIH/ y almacena el resultado de cada uno en .data/parsed_documents. En caso de que el argumento no coincida con ningún nombre entre los documentos en .data/DocsVIH, se muestra un warning indicando el documento no encontrado y se continúa el procesado del resto de documentos. 
- [Modulo App](./app.py): es el front sera la app de steamlit, una o varias ya veremos depende de la prueba
- [Modulo Info retrival](./semantic_db.py) : Elige entre varios modelos de embeddigns para crear una base de datos semantica apartir de los chunks generados por [chunker.py](./chunker.py). Puede hacer consultas a esta base de datos que retornen los chunks más cercanos.
- [Modulo Chunker](./chunker.py): Divide el texto de entrada en chunks por secciones. Si una sección es más grande que el tamaño máximo de chunk, la divide en varios chunks de menor tamaño.
- Modulo LLM: [A futuro]: aqui sera ya el promting al LLM

## Parametros 

### Función de cálculo de similaridad entre documentos
Fijaria la distancia coseno para calcular distancias en embeddings. 

### Estrategia de división en chunks
En el [módulo chunker](./chunker.py) se definen las distintas estrategias para dividir el texto en chunks. Las estrategias implementadas son:
 - División por secciones naturales y chunks de tamaño fijo
Esta estrategia divide el texto respetando sus secciones para conservar la coherecia y la relación semántica en el texto de cada sección. Se define un tamaño de chunk fijo, de forma que si una sección excede el tamaño de chunk se divide en varios chunks.

La estrategia por defecto es la división por secciones naturales y chunks de tamaño fijo.
### Modelos de embedding:
Los modelos de embedding utilizados son:
 - [Multilingual-E5-large](https://huggingface.co/intfloat/multilingual-e5-large): Es un modelo multilingüe que utiliza embeddings de 1024 tokens. Está entrenado en diversidad de fuentes (Wikipedia, Reddit, Stackxchange...) presentado en [Liang Wang et al., 2024](https://doi.org/10.48550/arXiv.2402.05672).
 - [Jina AI](https://huggingface.co/jinaai/jina-embeddings-v2-base-es): Es un modelo de embeddings multilingüe (español, inglés y alemán) propuesto por [Isabelle Mohr et al., 2024](https://doi.org/10.48550/arXiv.2402.17016).
### Modelo LLM y prompting
- llm
- prompt


## Front 


- caja seleccion modelo emebdding
- caja seleccion tipo de chunking

- caja para escribir la query 

- resultado mostrar los K chunks mas relvantes a la query

## Evaluacion 

3 fases 

1. Definir tecnica de chunk
2. Definir modelos embeddings
3. Elegir LLM 