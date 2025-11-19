from typing import Dict, Any, List, Optional
import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("⚠ Warning: tavily-python not installed. Web search will be unavailable.")


class WebSearch:

    def __init__(self):
        self.client = None
        self.llm = None
        self.setup_tavily()
        self.setup_llm()
        
    def setup_tavily(self):
        if not TAVILY_AVAILABLE:
            print("⚠ Tavily not available")
            return
            
        if config.TAVILY_API_KEY:
            try:
                self.client = TavilyClient(api_key=config.TAVILY_API_KEY)
                print("✓ Tavily web search configured")
            except Exception as e:
                print(f"⚠ Error configuring Tavily: {e}")
        else:
            print("⚠ Warning: TAVILY_API_KEY not set")
    
    def setup_llm(self):
        if config.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                openai_api_key=config.OPENAI_API_KEY,
                max_tokens=config.MAX_TOKENS
            )
        else:
            self.llm = None
    
    def search(self, query: str, max_results: int = config.MAX_WEB_RESULTS) -> Dict[str, Any]:
        if not self.client:
            return {
                'success': False,
                'error': 'Web search not configured. Please set TAVILY_API_KEY.',
                'results': []
            }
        
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=True,
                include_raw_content=False
            )
            results = []
            for item in response.get('results', []):
                results.append({
                    'title': item.get('title', 'Unknown'),
                    'url': item.get('url', ''),
                    'content': item.get('content', ''),
                    'score': item.get('score', 0.0)
                })
            
            return {
                'success': True,
                'query': query,
                'answer': response.get('answer', ''),
                'results': results
            }
            
        except Exception as e:
            print(f"✗ Error performing web search: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def format_web_context(self, search_results: Dict[str, Any]) -> str:
        if not search_results.get('success') or not search_results.get('results'):
            return "No web search results available."
        
        context_parts = []
        
        for i, result in enumerate(search_results['results'], 1):
            context_parts.append(
                f"[Web Source {i}]\n"
                f"Title: {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Content: {result['content']}\n"
                f"Relevance Score: {result['score']:.3f}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def generate_web_response(self, query: str, search_results: Dict[str, Any]) -> Dict[str, Any]:
        if not search_results.get('success'):
            return {
                'response': f"Web search failed: {search_results.get('error', 'Unknown error')}",
                'sources': [],
                'source_type': 'error'
            }
        
        if not self.llm:
            return {
                'response': search_results.get('answer', 'No answer available from web search.'),
                'sources': search_results.get('results', []),
                'source_type': 'web_search'
            }

        context = self.format_web_context(search_results)

        template = """You are a helpful assistant. Answer the user's question based on the web search results provided.

            Web Search Results:
            {context}

            User Question: {question}

            Instructions:
            1. Answer the question based on the web search results above
            2. Be concise and accurate
            3. Cite which sources you used (e.g., "According to Web Source 1...")
            4. Maintain the language of the question (if asked in Malay, respond in Malay)
            5. If the results don't fully answer the question, say so

            Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        try:
            response_text = chain.invoke({
                "context": context,
                "question": query
            })
            
            return {
                'response': response_text,
                'sources': search_results.get('results', []),
                'source_type': 'web_search',
                'query': query
            }
            
        except Exception as e:
            print(f"✗ Error generating web response: {e}")
            return {
                'response': f'Error generating response: {str(e)}',
                'sources': search_results.get('results', []),
                'source_type': 'error'
            }
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        search_results = self.search(query)
        result = self.generate_web_response(query, search_results)
        
        return result
