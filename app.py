import streamlit as st
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_processor import DataProcessor
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.web_search import WebSearch
from src.stats_analyzer import StatsAnalyzer
from src.ui_components import (
    display_message, display_sidebar, display_welcome_message,
    display_stats_result, display_quality_indicator
)


# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_system():
    """
    Initialize all system components (cached to avoid reloading).
    
    Returns:
        Tuple of (data_processor, vector_store, rag_engine, web_search, stats_analyzer)
    """
    with st.spinner("🔄 Initializing system..."):
        # Load data
        data_processor = DataProcessor()
        df = data_processor.load_data()
        
        # Initialize vector store
        vector_store = VectorStore()
        vector_store.initialize_db()
        
        # Check if we need to populate the vector store
        if vector_store.index is None or vector_store.index.ntotal == 0:
            st.info("📥 First-time setup: Building vector database. This may take a few minutes...")
            documents = data_processor.prepare_documents()
            vector_store.add_documents(documents)
            st.success("✅ Vector database built successfully!")
        
        # Initialize RAG engine
        rag_engine = RAGEngine(vector_store)
        
        # Initialize web search
        web_search = WebSearch()
        
        # Initialize stats analyzer
        stats_analyzer = StatsAnalyzer(df)
        
        return data_processor, vector_store, rag_engine, web_search, stats_analyzer


def handle_news_query(query: str, rag_engine: RAGEngine, web_search: WebSearch):
    """
    Handle a news query using RAG and web search fallback.
    
    Args:
        query: User query
        rag_engine: RAG engine instance
        web_search: Web search instance
    """
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process query with RAG
    with st.spinner("🔍 Searching knowledge base..."):
        rag_result = rag_engine.answer_query(query)
    
    # Check if we need web search
    quality = rag_result.get('quality_assessment', {})
    
    if quality.get('needs_web_search'):
        # Try web search
        with st.spinner("🌐 Searching the web..."):
            web_result = web_search.answer_query(query)
        
        # Display both results
        with st.chat_message("assistant"):
            # Show quality indicator
            display_quality_indicator(quality)
            
            # Show RAG result first (if any)
            if rag_result.get('sources'):
                st.markdown("### From Knowledge Base:")
                st.markdown(rag_result['response'])
                display_message("assistant", "", rag_result.get('sources', []), 'knowledge_base')
            
            # Show web search result
            st.markdown("### From Web Search:")
            st.markdown(web_result['response'])
            display_message("assistant", "", web_result.get('sources', []), 'web_search')
            
            # Combine for history
            combined_response = f"**From Knowledge Base:**\n{rag_result['response']}\n\n**From Web Search:**\n{web_result['response']}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": combined_response,
                "sources": rag_result.get('sources', []) + web_result.get('sources', []),
                "source_type": "combined"
            })
    else:
        # Display RAG result only
        with st.chat_message("assistant"):
            display_quality_indicator(quality)
            st.markdown(rag_result['response'])
            display_message("assistant", "", rag_result.get('sources', []), 'knowledge_base')
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": rag_result['response'],
            "sources": rag_result.get('sources', []),
            "source_type": rag_result.get('source_type', 'knowledge_base')
        })


def handle_stats_query(query: str, stats_analyzer: StatsAnalyzer):
    """
    Handle a statistics query.
    
    Args:
        query: User query
        stats_analyzer: Stats analyzer instance
    """
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Add to chat history
    st.session_state.stats_messages.append({"role": "user", "content": query})
    
    # Process query
    with st.spinner("📊 Analyzing dataset..."):
        result = stats_analyzer.answer_query(query)
    
    # Display result
    with st.chat_message("assistant"):
        display_stats_result(result)
    
    # Add to history
    st.session_state.stats_messages.append({
        "role": "assistant",
        "result": result
    })


def main():
    """Main application function."""
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'stats_messages' not in st.session_state:
        st.session_state.stats_messages = []
    
    # Display sidebar and get settings
    mode, top_k, temperature = display_sidebar()
    
    # Update config with user settings
    config.TOP_K_RESULTS = top_k
    config.TEMPERATURE = temperature
    
    # Initialize system
    try:
        data_processor, vector_store, rag_engine, web_search, stats_analyzer = initialize_system()
    except Exception as e:
        st.error(f"❌ Error initializing system: {e}")
        st.info("Please make sure you have set up your .env file with the required API keys.")
        st.stop()
    
    # Main content area
    if mode == "📰 News Query":
        st.title("📰 News Query Assistant")
        st.markdown("Ask questions about news from our knowledge base or the web.")
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                    if message.get("sources") and message.get("source_type") != "combined":
                        display_message("assistant", "", message.get("sources", []), message.get("source_type"))
        
        # Chat input
        if prompt := st.chat_input("Ask a question about news..."):
            handle_news_query(prompt, rag_engine, web_search)
        
        # Show example questions if no messages
        if not st.session_state.messages:
            st.markdown("### 💡 Example Questions:")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("What are some initiatives launched by MCMC?"):
                    handle_news_query("What are some initiatives launched by MCMC?", rag_engine, web_search)
                    st.rerun()
            
            with col2:
                if st.button("Adakah SSM terbabit dengan kes-kes mahkamah?"):
                    handle_news_query("Adakah SSM terbabit dengan kes-kes mahkamah?", rag_engine, web_search)
                    st.rerun()
    
    else:  # Dataset Statistics mode
        st.title("📊 Dataset Statistics")
        st.markdown("Query statistics about the news dataset using natural language.")
        
        # Show quick stats
        with st.expander("📈 Quick Overview", expanded=True):
            quick_stats = stats_analyzer.get_quick_stats()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Articles", quick_stats['total_articles'])
            
            with col2:
                sentiment_dist = quick_stats.get('sentiment_distribution', {})
                if sentiment_dist:
                    st.markdown("**Sentiment Distribution:**")
                    for sentiment, count in sentiment_dist.items():
                        st.markdown(f"- {sentiment}: {count}")
            
            with col3:
                date_range = quick_stats.get('date_range', {})
                st.markdown("**Date Range:**")
                st.markdown(f"- Earliest: {date_range.get('earliest', 'N/A')}")
                st.markdown(f"- Latest: {date_range.get('latest', 'N/A')}")
        
        # Display stats chat history
        for message in st.session_state.stats_messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    display_stats_result(message["result"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about dataset statistics..."):
            handle_stats_query(prompt, stats_analyzer)
        
        # Show example questions if no messages
        if not st.session_state.stats_messages:
            st.markdown("### 💡 Example Questions:")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("How many positive and negative news are there?"):
                    handle_stats_query("How many positive and negative news are there?", stats_analyzer)
                    st.rerun()
            
            with col2:
                if st.button("How many news articles are before June 2025?"):
                    handle_stats_query("How many news articles are before June 2025?", stats_analyzer)
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "AI News Assistant | Powered by OpenAI & RAG"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
