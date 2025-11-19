from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config
from src.vector_store import VectorStore


class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = None
        self.setup_llm()
        
    def setup_llm(self):
        print("Initializing RAG Engine...")
        if config.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                openai_api_key=config.OPENAI_API_KEY,
                max_tokens=config.MAX_TOKENS
            )
            print("✓ RAG Engine initialized with OpenAI")
        else:
            print("⚠ Warning: OPENAI_API_KEY not set")
    
    def retrieve_context(self, query: str, top_k: int = config.TOP_K_RESULTS) -> Dict[str, Any]:
        print('Retrieving context from vector store')
        return self.vector_store.search(query, top_k=top_k)
    
    def assess_retrieval_quality(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        if not search_results['results']:
            return {
                'has_relevant_results': False,
                'best_similarity': 0.0,
                'needs_web_search': True,
                'reason': 'No results found in knowledge base'
            }
        
        best_similarity = search_results['results'][0]['similarity']

        has_relevant = best_similarity >= config.SIMILARITY_THRESHOLD
        needs_web_search = best_similarity < config.WEB_SEARCH_THRESHOLD
        print(has_relevant, best_similarity, needs_web_search)
        return {
            'has_relevant_results': has_relevant,
            'best_similarity': best_similarity,
            'needs_web_search': needs_web_search,
            'reason': f"Best similarity: {best_similarity:.3f}"
        }
    
    def format_context(self, search_results: Dict[str, Any]) -> str:
        if not search_results['results']:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        
        for i, result in enumerate(search_results['results'], 1):
            metadata = result['metadata']
            
            title = metadata.get('title', 'Unknown')
            summary = metadata.get('summary', '')
            author = metadata.get('author', 'Unknown')
            sentiment = metadata.get('sentiment', 'Neutral')
            timestamp = metadata.get('timestamp', 'Unknown')
            
            context_parts.append(
                f"[Source {i}]\n"
                f"Title: {title}\n"
                f"Author/Source: {author}\n"
                f"Timestamp: {timestamp}\n"
                f"Sentiment: {sentiment}\n"
                f"Summary: {summary}\n"
                f"Similarity Score: {result['similarity']:.3f}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def generate_response(self, query: str, context: str, 
                         search_results: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm:
            return {
                'response': 'Error: OpenAI API not configured',
                'sources': [],
                'source_type': 'error'
            }
        
        # Create prompt template
        template = """You are a helpful news assistant. Answer the user's question based on the provided news articles from the knowledge base.

            Context from News Articles:
            {context}

            User Question: {question}

            Instructions:
            1. Answer the question based ONLY on the information provided in the context above
            2. If the context doesn't contain enough information to answer fully, say so
            3. Be concise and accurate
            4. Cite which sources you used (e.g., "According to Source 1...")
            5. Maintain the language of the question (if asked in Malay, respond in Malay)

            Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        try:
            response_text = chain.invoke({
                "context": context,
                "question": query
            })

            sources = []
            for result in search_results['results']:
                metadata = result['metadata']
                sources.append({
                    'title': metadata.get('title', 'Unknown'),
                    'author': metadata.get('author', 'Unknown'),
                    'timestamp': metadata.get('timestamp', 'Unknown'),
                    'sentiment': metadata.get('sentiment', 'Neutral'),
                    'url': metadata.get('url', ''),
                    'similarity': result['similarity']
                })
            
            return {
                'response': response_text,
                'sources': sources,
                'source_type': 'knowledge_base',
                'query': query
            }
            
        except Exception as e:
            print(f"✗ Error generating response: {e}")
            return {
                'response': f'Error generating response: {str(e)}',
                'sources': [],
                'source_type': 'error'
            }
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        search_results = self.retrieve_context(query)
        quality = self.assess_retrieval_quality(search_results)
        context = self.format_context(search_results)
        result = self.generate_response(query, context, search_results)
        result['quality_assessment'] = quality
        
        return result
