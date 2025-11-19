
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.7
MAX_TOKENS = 2048

TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.3  
WEB_SEARCH_THRESHOLD = 0.5  

VECTOR_DB_PATH = "./data/faiss_index"
METADATA_PATH = "./data/metadata.pkl"
COLLECTION_NAME = "news_articles"

NEWS_CSV_PATH = "./news.csv"
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200  

MAX_WEB_RESULTS = 5

PAGE_TITLE = "AI News Assistant"
PAGE_ICON = "📰"
