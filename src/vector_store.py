"""
Vector store management using FAISS and LangChain for semantic search.
"""
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
import config


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = [] 
        self.embeddings = None
        self.setup_embeddings()
        
    def setup_embeddings(self):
        if config.OPENAI_API_KEY:
            self.embeddings = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL,
                openai_api_key=config.OPENAI_API_KEY
            )
            print("✓ OpenAI Embeddings configured")
        else:
            print("⚠ Warning: OPENAI_API_KEY not set")
    
    def initialize_db(self):
        try:
            if os.path.exists(config.VECTOR_DB_PATH) and os.path.exists(config.METADATA_PATH):
                print(f"Loading FAISS index from {config.VECTOR_DB_PATH}...")
                self.index = faiss.read_index(config.VECTOR_DB_PATH)
                
                print(f"Loading metadata from {config.METADATA_PATH}...")
                with open(config.METADATA_PATH, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                print(f"✓ Vector database loaded ({self.index.ntotal} documents)")
            else:
                print("Creating new FAISS index...")
                self.index = None
                self.metadata = []
                os.makedirs(os.path.dirname(config.VECTOR_DB_PATH), exist_ok=True)
                
        except Exception as e:
            print(f"✗ Error initializing vector database: {e}")
            raise
    
    def save_db(self):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            return
            
        try:
            faiss.write_index(self.index, config.VECTOR_DB_PATH)
            with open(config.METADATA_PATH, 'wb') as f:
                pickle.dump(self.metadata, f)
                
            print(f"✓ Vector database saved to {config.VECTOR_DB_PATH}")
            
        except Exception as e:
            print(f"✗ Error saving vector database: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:

        if not self.embeddings:
            raise ValueError("OpenAI Embeddings not configured")
            
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"✗ Error generating embedding: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        if not self.embeddings:
            print("⚠ Cannot add documents: Embeddings not configured")
            return

        total = len(documents)
        print(f"Adding {total} documents to vector store...")
        
        all_embeddings = []
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc['text'] for doc in batch]
            
            try:
                batch_embeddings = self.embeddings.embed_documents(texts)
                all_embeddings.extend(batch_embeddings)

                for j, doc in enumerate(batch):
                    self.metadata.append({
                        'text': doc['text'],
                        'metadata': doc['metadata']
                    })
                
                print(f"  Processed batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"✗ Error processing batch {i}: {e}")
                continue
        
        if not all_embeddings:
            print("⚠ No embeddings generated")
            return

        embeddings_array = np.array(all_embeddings).astype('float32')

        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            print(f"✓ Created new FAISS index with dimension {dimension}")

        self.index.add(embeddings_array)

        self.save_db()
        print(f"✓ Successfully added {total} documents to vector store")
    
    def search(self, query: str, top_k: int = config.TOP_K_RESULTS) -> Dict[str, Any]:
        if self.index is None:
            self.initialize_db()
            if self.index is None or self.index.ntotal == 0:
                return {'query': query, 'results': []}
        
        try:
            query_embedding = self.generate_embedding(query)
            query_vector = np.array([query_embedding]).astype('float32')

            distances, indices = self.index.search(query_vector, top_k)
    
            formatted_results = {
                'query': query,
                'results': []
            }
            
            for i in range(len(indices[0])):
                idx = indices[0][i]
                distance = distances[0][i]
                
                if idx == -1:
                    continue

                if idx < len(self.metadata):
                    doc_data = self.metadata[idx]
                    
                    similarity = 1 / (1 + distance)
                    
                    formatted_results['results'].append({
                        'id': str(idx),
                        'text': doc_data['text'],
                        'metadata': doc_data['metadata'],
                        'similarity': similarity,
                        'distance': float(distance)
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"✗ Error searching vector store: {e}")
            raise
    
    def reset_database(self):
        if os.path.exists(config.VECTOR_DB_PATH):
            os.remove(config.VECTOR_DB_PATH)
        if os.path.exists(config.METADATA_PATH):
            os.remove(config.METADATA_PATH)
            
        self.index = None
        self.metadata = []
        print("✓ Reset vector database")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        if self.index is None:
            self.initialize_db()
            
        count = self.index.ntotal if self.index else 0
        
        return {
            'name': config.COLLECTION_NAME,
            'count': count,
            'type': 'FAISS IndexFlatL2'
        }
