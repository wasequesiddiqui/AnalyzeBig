
# %% [markdown]
# # RAG retrievers with watsonx.ai + LangChain
# Required env vars (.env): WATSONX_URL, WATSONX_APIKEY, WATSONX_PROJECT_ID
# pip install ibm-watsonx-ai langchain langchain-community langchain-ibm
#             langchain-text-splitters chromadb lark pypdf python-dotenv
 
# %% Imports
import logging
import os
import urllib.request
from functools import lru_cache
 
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
load_dotenv(override=True)
 
# %% Config
WATSONX_URL = os.environ["WATSONX_URL"]
WATSONX_APIKEY = os.environ["WATSONX_APIKEY"]  # missing before -> WMLClientError
WATSONX_PROJECT_ID = os.environ["WATSONX_PROJECT_ID"]
 
LLM_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"
EMBED_MODEL_ID = "ibm/granite-embedding-278m-multilingual"
 
 
# %% Model factories
@lru_cache(maxsize=1)
def get_llm() -> WatsonxLLM:
    """Create the LLM once and reuse it."""
    return WatsonxLLM(
        model_id=LLM_MODEL_ID,
        url=WATSONX_URL,
        apikey=WATSONX_APIKEY,
        project_id=WATSONX_PROJECT_ID,
        params={
            GenParams.MAX_NEW_TOKENS: 256,  # max tokens in the output
            GenParams.TEMPERATURE: 0.5,     # creativity of the response
        },
    )
 
 
@lru_cache(maxsize=1)
def get_embeddings() -> WatsonxEmbeddings:
    """Create the embedding model once and reuse it."""
    return WatsonxEmbeddings(
        model_id=EMBED_MODEL_ID,
        url=WATSONX_URL,
        apikey=WATSONX_APIKEY,
        project_id=WATSONX_PROJECT_ID,
        params={
            # Was 3 before, which cut every chunk down to 3 tokens.
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        },
    )
 
 
# %% Helpers
def split_documents(data, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(data)
 
 
def reset_vectordb(vectordb: Chroma) -> None:
    """Delete everything stored in the vector DB."""
    ids = vectordb.get()["ids"]
    if ids:
        vectordb.delete(ids)
 
 
# %% 1. Load a text file and build a vector store
URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "MZ9z1lm-Ui3YBp3SYWLTAQ/companypolicies.txt"
)
TXT_FILE = "companypolicies.txt"
 
if not os.path.exists(TXT_FILE):
    print(f"Downloading {URL}...")
    urllib.request.urlretrieve(URL, TXT_FILE)
    print(f"Saved as '{TXT_FILE}'")
 
txt_data = TextLoader(TXT_FILE).load()
chunks_txt = split_documents(txt_data, chunk_size=200, chunk_overlap=20)
vectordb = Chroma.from_documents(chunks_txt, get_embeddings())
 
# %% 2. Basic retriever variants
query = "email policy"
 
# default similarity search
docs = vectordb.as_retriever().invoke(query)
print("similarity:", len(docs))
 
# top-1 only
docs = vectordb.as_retriever(search_kwargs={"k": 1}).invoke(query)
print("k=1:", docs)
 
# maximal marginal relevance (diverse results)
docs = vectordb.as_retriever(search_type="mmr").invoke(query)
print("mmr:", len(docs))
 
# similarity with score threshold
docs = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4},
).invoke(query)
print("threshold:", len(docs))
 
# %% 3. Load a PDF and rebuild the vector store
pdf_data = PyPDFLoader(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf"
).load()
chunks_pdf = split_documents(pdf_data, chunk_size=500, chunk_overlap=20)
 
reset_vectordb(vectordb)  # drop the previous document's embeddings
vectordb = Chroma.from_documents(chunks_pdf, get_embeddings())
 
# %% 4. Multi-query retriever
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
 
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=get_llm(),
)
docs = multi_query_retriever.invoke("What does the paper say about langchain?")
print("multi-query:", len(docs))
 
# %% 5. Self-query retriever (movie example)
movie_docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
 
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="director", description="The name of the movie director", type="string"),
    AttributeInfo(name="rating", description="A 1-10 rating for the movie", type="float"),
]
 
movie_vectordb = Chroma.from_documents(movie_docs, get_embeddings())
#%%
from langchain_community.query_constructors.chroma import ChromaTranslator

self_query_retriever = SelfQueryRetriever.from_llm(
    get_llm(),
    movie_vectordb,
    "Brief summary of a movie.",
    metadata_field_info,
    structured_query_translator=ChromaTranslator(),
)
 
# %% Try it
print(self_query_retriever.invoke("Movies rated higher than 8.5"))
# %%
# This example specifies a query and a filter
print(self_query_retriever.invoke("Has Greta Gerwig directed any movies about women"))
# %%
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain.storage import InMemoryStore
# %%
# Set two splitters. One is with big chunk size (parent) and one is with small chunk size (child)
parent_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separator='\n')
child_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator='\n')
# %%
vectordb = Chroma(
    collection_name="split_parents", embedding_function=watsonx_embedding()
)
#vectordb = Chroma.from_documents(documents=chunks_pdf, embedding=watsonx_embedding())
# The storage layer for the parent documents
store = InMemoryStore()
# %%
retriever = ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
# %%
retriever.add_documents(txt_data)
# %%
len(list(store.yield_keys()))
# %%
sub_docs = vectordb.similarity_search("smoking policy")
# %%
print(sub_docs[0].page_content)
# %%
retrieved_docs = retriever.invoke("smoking policy")
print(retrieved_docs[0].page_content)
# %%
