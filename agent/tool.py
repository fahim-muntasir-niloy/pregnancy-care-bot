from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from exa_py import Exa

from langchain_core.tools import tool
# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")



# Initialize embedding model
model_name = "BAAI/bge-m3"
encode_kwargs = {"normalize_embeddings": True}

hf_embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    show_progress=True,
    encode_kwargs=encode_kwargs
)

# Initialize vector store
vector_store = AstraDBVectorStore(
    embedding=hf_embedding_model,
    collection_name="pregnancy_bot",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace='default_keyspace',
    autodetect_collection=True
)

# Initialize retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3, "score_threshold": 0.5}
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Define the RAG tool
@tool
def retrieve_relevant_info(query: str) -> str:
    """Retrieve relevant information from the knowledge base about pregnancy.
    
    Args:
        query: The query to retrieve relevant information from the knowledge base.
        
    Returns:
        The relevant information from the knowledge base.
    """
    return retriever.invoke(query)


exa = Exa(api_key=os.environ["EXA_API_KEY"])


@tool
def search_web(query: str) -> str:
  """Search for webpages based on the query and retrieve their contents."""
  return exa.search_and_contents(query, text = True)

