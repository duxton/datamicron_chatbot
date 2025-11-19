import pandas as pd
import re
from typing import List, Dict, Any
import config


class DataProcessor:
    
    def __init__(self, csv_path: str = config.NEWS_CSV_PATH):
        self.csv_path = csv_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            print(f"✓ Loaded {len(self.df)} news articles from {self.csv_path}")
            return self.df
        except Exception as e:
            print(f"✗ Error loading CSV: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def prepare_documents(self) -> List[Dict[str, Any]]:
        if self.df is None:
            self.load_data()
        
        documents = []
        
        for idx, row in self.df.iterrows():
            title = self.clean_text(row.get('title', ''))
            summary = self.clean_text(row.get('summary', ''))
            content = self.clean_text(row.get('article_content', ''))
            

            full_text = f"Title: {title}\n\nSummary: {summary}\n\nContent: {content}"
            
            metadata = {
                'id': str(row.get('id', idx)),
                'title': title,
                'summary': summary,
                'author': self.clean_text(row.get('author', 'Unknown')),
                'sentiment': self.clean_text(row.get('sentiment', 'Neutral')),
                'timestamp': self.clean_text(row.get('timestamp', '')),
                'url': self.clean_text(row.get('url', '')),
                'source_country': self.clean_text(row.get('source_country', '')),
            }
            
            documents.append({
                'text': full_text,
                'metadata': metadata
            })
        
        print(f"✓ Prepared {len(documents)} documents for embedding")
        return documents
    
    def get_dataframe(self) -> pd.DataFrame:
        if self.df is None:
            self.load_data()
        return self.df
    
    def chunk_text(self, text: str, chunk_size: int = config.CHUNK_SIZE, 
                   overlap: int = config.CHUNK_OVERLAP) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
