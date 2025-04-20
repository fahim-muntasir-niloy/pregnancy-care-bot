import os

from langchain_astradb import AstraDBVectorStore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

# credentials
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

os.environ['LANGSMITH_TRACING_V2']="true"
os.environ['LANGSMITH_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGSMITH_PROJECT']="pregnancy_bot"

# embedding model
model_name = "BAAI/bge-m3"
encode_kwargs = {"normalize_embeddings": True}

hf_embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    show_progress=True,
    encode_kwargs=encode_kwargs
)

# as the vector data is already stored in Astra DB
vector_store = AstraDBVectorStore(
    embedding=hf_embedding_model,
    collection_name="pregnancy_bot",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace='default_keyspace',
    autodetect_collection=True
)

retriever = vector_store.as_retriever(search_type="similarity", 
                                     search_kwargs={"k": 3, 
                                                    "score_threshold": 0.5,
                                    }
                                )

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


# format context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# RAG-Chain
template = """You are a doctor who is answering pregnancy questions.
Use the following pieces of context to answer the question at the end.
Always reply in bangla text, do not answer in english. No need to translate in english too.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use ten sentences maximum and keep the answer as concise as possible.
End the answer on a single । punctuation.

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)
