import streamlit as st
from typing import Dict, Any, List


def display_message(role: str, content: str, sources: List[Dict[str, Any]] = None, 
                   source_type: str = None):
    with st.chat_message(role):
        st.markdown(content)
        
        if sources and role == 'assistant':
            if source_type == 'knowledge_base':
                display_kb_sources(sources)
            elif source_type == 'web_search':
                display_web_sources(sources)


def display_kb_sources(sources: List[Dict[str, Any]]):
    if not sources:
        return
    
    st.markdown("---")
    st.markdown("**📚 Sources from Knowledge Base:**")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source.get('title', 'Unknown')}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Author/Source:** {source.get('author', 'Unknown')}")
                st.markdown(f"**Timestamp:** {source.get('timestamp', 'Unknown')}")
                
                if source.get('url'):
                    st.markdown(f"**URL:** [{source['url']}]({source['url']})")
            
            with col2:
                sentiment = source.get('sentiment', 'Neutral')
                sentiment_emoji = {
                    'Positive': '😊',
                    'Negative': '😞',
                    'Neutral': '😐'
                }.get(sentiment, '😐')
                
                st.markdown(f"**Sentiment:** {sentiment_emoji} {sentiment}")
                
                similarity = source.get('similarity', 0)
                st.markdown(f"**Relevance:** {similarity:.1%}")


def display_web_sources(sources: List[Dict[str, Any]]):
    if not sources:
        return
    
    st.markdown("---")
    st.markdown("**🌐 Sources from Web Search:**")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Web Source {i}: {source.get('title', 'Unknown')}", expanded=False):
            st.markdown(f"**URL:** [{source.get('url', 'N/A')}]({source.get('url', '#')})")
            
            if source.get('content'):
                st.markdown(f"**Snippet:** {source['content'][:300]}...")
            
            if source.get('score'):
                st.markdown(f"**Relevance Score:** {source['score']:.1%}")


def display_stats_result(result: Dict[str, Any]):
    if not result.get('success'):
        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        return
    
    st.success("✅ Query executed successfully!")

    if result.get('explanation'):
        st.info(f"**Analysis:** {result['explanation']}")
    
    # Show result
    st.markdown("**Result:**")
    formatted_result = result.get('formatted_result', '')
    
    # Try to display as a nice table if it's a Series or DataFrame
    if 'sentiment' in result.get('query', '').lower() or 'distribution' in result.get('query', '').lower():
        # Likely a value_counts result, display as chart
        try:
            import pandas as pd
            if isinstance(result.get('result'), pd.Series):
                st.bar_chart(result['result'])
        except:
            pass
    
    st.code(formatted_result, language='text')
    
    # Show the generated code in an expander
    if result.get('code'):
        with st.expander("🔍 View Generated Code", expanded=False):
            st.code(result['code'], language='python')


def display_sidebar():
    """Display sidebar with settings and information."""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["📰 News Query", "📊 Dataset Statistics"],
            help="Choose between asking questions about news or querying dataset statistics"
        )
        
        st.markdown("---")
        
        # API Key status
        st.markdown("### 🔑 API Status")
        
        import config
        
        if config.OPENAI_API_KEY:
            st.success("✅ OpenAI API configured")
        else:
            st.error("❌ OpenAI API not configured")
            st.info("Set OPENAI_API_KEY in .env file")
        
        if config.TAVILY_API_KEY:
            st.success("✅ Tavily API configured")
        else:
            st.warning("⚠️ Tavily API not configured (web search unavailable)")
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            top_k = st.slider(
                "Number of results to retrieve",
                min_value=1,
                max_value=10,
                value=config.TOP_K_RESULTS,
                help="How many relevant documents to retrieve from the knowledge base"
            )
            
            temperature = st.slider(
                "Response creativity (temperature)",
                min_value=0.0,
                max_value=1.0,
                value=config.TEMPERATURE,
                step=0.1,
                help="Higher values make responses more creative, lower values more focused"
            )
            
            return mode, top_k, temperature
        
        return mode, config.TOP_K_RESULTS, config.TEMPERATURE


def display_welcome_message():
    """Display welcome message and instructions."""
    st.markdown("""
    # 📰 AI News Assistant
    
    Welcome! I can help you with:
    
    1. **📰 News Queries**: Ask questions about news from our knowledge base
       - Example: "What are some initiatives launched by MCMC?"
       - Example: "Adakah SSM terbabit dengan kes-kes mahkamah?"
    
    2. **📊 Dataset Statistics**: Query statistics about the news dataset
       - Example: "How many positive and negative news are there?"
       - Example: "How many news articles are before June 2025?"
    
    Select a mode from the sidebar and start asking questions!
    """)


def display_quality_indicator(quality: Dict[str, Any]):
    """
    Display quality assessment indicator.
    
    Args:
        quality: Quality assessment dictionary
    """
    if quality.get('needs_web_search'):
        st.info(f"ℹ️ Knowledge base results limited (similarity: {quality.get('best_similarity', 0):.1%}). Using web search for additional information.")
    elif quality.get('has_relevant_results'):
        st.success(f"✅ Found relevant information in knowledge base (similarity: {quality.get('best_similarity', 0):.1%})")
    else:
        st.warning("⚠️ No relevant results found in knowledge base")
