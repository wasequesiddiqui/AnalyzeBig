#%%
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

import wget

# LangChain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# IBM watsonx
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import (
    ModelTypes,
    DecodingMethods,
)
from langchain_ibm import WatsonxLLM

print("All imports successful!")

# %%
filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

# Use wget to download the file
wget.download(url, out=filename)
print('file downloaded')
# %%
with open(filename, 'r') as file:
    # Read the contents of the file
    contents = file.read()
    print(contents)
# %%
loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(len(texts))
# %%
embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)  # store the embedding in docsearch using Chromadb
print('document ingested')
# %%
model_id  = 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
# %%
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
    GenParams.MIN_NEW_TOKENS: 130, # this controls the minimum number of tokens in the generated output
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
}
# %%
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# %%
llm = WatsonxLLM(
    model_id=model_id,
    url="https://eu-de.ml.cloud.ibm.com",
    project_id=os.environ["WATSONX_PROJECT_ID"],
    params=parameters,
)
#%%
qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 return_source_documents=False)
query = "what is mobile policy?"
qa.invoke(query)

# %%
qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 return_source_documents=False)
query = "Can you summarize the document for me?"
qa.invoke(query)

# %%
qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 return_source_documents=False)
query = "Can I eat in company vehicles?"
qa.invoke(query)

# %%
prompt_template = """Use the information from the document to answer the question at the end. If you don't know the answer, just say that you don't know, definitely do not try to make up an answer.

{context}

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

# %%
qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 chain_type_kwargs=chain_type_kwargs, 
                                 return_source_documents=False)

query = "Can I eat in company vehicles?"
qa.invoke(query)

# %%
query = "What I cannot do in it?"
qa.invoke(query)

# %%
memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)

# %%
qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                           chain_type="stuff", 
                                           retriever=docsearch.as_retriever(), 
                                           memory = memory, 
                                           get_chat_history=lambda h : h, 
                                           return_source_documents=False)

#%%
history = []

# %%
query = "What is mobile policy?"
result = qa.invoke({"question":query}, {"chat_history": history})
print(result["answer"])

# %%
history.append((query, result["answer"]))

# %%
query = "List points in it?"
result = qa({"question": query}, {"chat_history": history})
print(result["answer"])

# %%
history.append((query, result["answer"]))

# %%
query = "What is the aim of it?"
result = qa({"question": query}, {"chat_history": history})
print(result["answer"])

# %%
def qa():
    memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                               chain_type="stuff", 
                                               retriever=docsearch.as_retriever(), 
                                               memory = memory, 
                                               get_chat_history=lambda h : h, 
                                               return_source_documents=False)
    history = []
    while True:
        query = input("Question: ")
        
        if query.lower() in ["quit","exit","bye"]:
            print("Answer: Goodbye!")
            break
            
        result = qa({"question": query}, {"chat_history": history})
        
        history.append((query, result["answer"]))
        
        print("Answer: ", result["answer"])

# %%
qa()
# %%
