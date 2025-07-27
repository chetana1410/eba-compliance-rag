"""
Regulation RAG Assistant - Working Streamlit UI
A user-friendly interface for querying regulatory documents using ColBERT RAG system.
"""

import streamlit as st
import asyncio
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, TypedDict
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define RAGResult type
class RAGResult(TypedDict):
    answer: str
    retrieved_articles: List[str]

# Page configuration
st.set_page_config(
    page_title="Regulation RAG Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-indicator {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .answer-container {
        background-color: #f8f9fa;
        color: #212529;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        line-height: 1.6;
    }
    .citation-item {
        background-color: #e9ecef;
        color: #212529;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Example questions for the regulatory domain
EXAMPLE_QUESTIONS = [
    "What are the capital requirements for credit risk?",
    "How is operational risk calculated under CRR?",
    "What are the liquidity coverage ratio requirements?",
    "What are the minimum capital requirements for banks?",
    "How are credit risk weights determined for different asset classes?",
    "What is the definition of Tier 1 capital?",
    "What are the requirements for internal ratings-based approaches?",
    "How is the leverage ratio calculated?",
]

def check_environment():
    """Check if the required environment is set up."""
    issues = []
    
    # Check environment variables
    if not os.getenv("OPENROUTER_API_KEY"):
        issues.append("OPENROUTER_API_KEY environment variable not set")
    
    if not os.getenv("DATA_PATH"):
        issues.append("DATA_PATH environment variable not set")
    
    # Check if Weaviate is running
    try:
        import requests
        response = requests.get("http://localhost:8080/v1/.well-known/ready", timeout=5)
        if response.status_code != 200:
            issues.append("Weaviate is not running or not accessible")
    except:
        issues.append("Weaviate is not running or not accessible")
    
    return len(issues) == 0, issues

def display_system_status():
    """Display system status in the sidebar."""
    st.sidebar.markdown("### 🔧 System Status")
    
    # Check system setup
    is_ready, issues = check_environment()
    
    if is_ready:
        st.sidebar.markdown(
            '<div class="status-indicator status-success">✅ System ready</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div class="status-indicator status-error">❌ System not ready</div>',
            unsafe_allow_html=True
        )
        
        st.sidebar.markdown("**Issues found:**")
        for issue in issues:
            st.sidebar.markdown(f"• {issue}")
        
        # Show troubleshooting tips
        st.sidebar.markdown("#### Troubleshooting:")
        st.sidebar.markdown("""
        - Ensure Weaviate is running: 
          ```bash
          docker run --detach -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.30.1
          ```
        - Set environment variables in `.env` file
        - Check that all dependencies are installed
        """)
    
    return is_ready

def display_configuration_panel():
    """Display configuration options in the sidebar."""
    st.sidebar.markdown("### ⚙️ Configuration")
    
    # Retrieval settings
    k_value = st.sidebar.slider(
        "Number of documents to retrieve",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        help="Higher values may provide more context but slower processing"
    )
    
    # Advanced settings in an expander
    with st.sidebar.expander("🔬 Advanced Settings"):
        st.markdown("**Model Configuration:**")
        st.text("• Embedding: lightonai/Reason-ModernColBERT")
        st.text("• HyDE: google/gemini-2.5-flash")
        st.text("• Answer: openai/gpt-4.1")
        
        st.markdown("**Features:**")
        st.text("• HyDE query enhancement: ✅")
        st.text("• Winnowing filter: ✅")
        st.text("• Multi-vector retrieval: ✅")
    
    return k_value

def display_example_questions():
    """Display example questions in the sidebar."""
    st.sidebar.markdown("### 💡 Example Questions")
    st.sidebar.markdown("Click on any question to use it:")
    
    # Initialize session state for selected question if not exists
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""
    
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        if st.sidebar.button(
            question,
            key=f"example_{i}",
            help="Click to use this question",
            use_container_width=True
        ):
            st.session_state.selected_question = question
            st.rerun()
    
    return st.session_state.selected_question

async def call_colbert_directly(question: str, k_value: int) -> RAGResult:
    """Call ColBERT system directly."""
    try:
        # Add paths
        current_dir = Path(".")
        src_path = current_dir / "src"
        sys.path.insert(0, str(src_path))
        
        from regulations_rag_eval.rag_implementations.ColBERT.generate_answers import generate_answers
        
        results = await generate_answers(
            questions=[question],
            implementation_name="ColBERT",
            params={"k": k_value}
        )
        
        if results and len(results) > 0:
            result = results[0]
            return RAGResult(
                answer=result.answer,
                retrieved_articles=result.retrieved_articles
            )
        else:
            return RAGResult(
                answer="No answer could be generated for this question.",
                retrieved_articles=[]
            )
    except Exception as e:
        return RAGResult(
            answer=f"An error occurred while processing your question: {str(e)}",
            retrieved_articles=[]
        )

async def process_question_async(question: str, k_value: int, system_ready: bool) -> RAGResult:
    """Process a question using the ColBERT system."""
    if not system_ready:
        # Return mock response if system not ready
        return await mock_process_question_async(question, k_value)
    
    # Try to call the actual ColBERT system directly
    try:
        result = await call_colbert_directly(question, k_value)
        return result
    except Exception as e:
        return RAGResult(
            answer=f"An error occurred while processing your question: {str(e)}",
            retrieved_articles=[]
        )

async def mock_process_question_async(question: str, k_value: int) -> RAGResult:
    """Mock function to simulate ColBERT processing."""
    # Simulate processing time
    await asyncio.sleep(1)
    
    # Mock response based on question content
    if ("capital requirements" in question.lower() or "credit risk" in question.lower() or 
        "reserves" in question.lower() or "shareholders" in question.lower() or 
        "article 26" in question.lower() or "tier 1" in question.lower()):
        answer = """
        According to the Capital Requirements Regulation (CRR), banks must maintain minimum capital requirements to cover credit risk, operational risk, and market risk. The key requirements include:

        1. **Common Equity Tier 1 (CET1) ratio**: Minimum 4.5% of risk-weighted assets
        2. **Tier 1 capital ratio**: Minimum 6% of risk-weighted assets  
        3. **Total capital ratio**: Minimum 8% of risk-weighted assets

        Additionally, banks may be subject to:
        - Capital conservation buffer: 2.5% of risk-weighted assets
        - Countercyclical capital buffer: 0-2.5% (varies by jurisdiction)
        - Systemic risk buffers for systemically important institutions

        These requirements ensure banks maintain adequate capital to absorb losses and continue operations during periods of stress.
        """
        citations = [
            "Article 92 (Minimum capital requirements)",
            "Article 25 (Common Equity Tier 1 items)",
            "Article 26 (Common Equity Tier 1 instruments)",
            "Article 129 (Capital conservation buffer)"
        ]
    elif "operational risk" in question.lower():
        answer = """
        Operational risk under CRR is calculated using standardized approaches that consider the bank's business activities and historical loss data. The main approaches are:

        1. **Basic Indicator Approach (BIA)**: 15% of average annual gross income over 3 years
        2. **Standardized Approach (TSA)**: Different percentages applied to gross income from different business lines
        3. **Advanced Measurement Approaches (AMA)**: Internal models based on loss data, scenario analysis, and business environment factors

        The operational risk capital requirement is calculated as 12.5 times the operational risk capital charge, which represents 8% capital requirement against operational risk exposure.
        """
        citations = [
            "Article 315 (Operational risk)",
            "Article 316 (Basic Indicator Approach)",
            "Article 317 (Standardized Approach)",
            "Article 321 (Advanced Measurement Approaches)"
        ]
    else:
        answer = f"""
        Thank you for your question about "{question}". 

        This is a demonstration of the Regulation RAG Assistant interface. In a fully operational system, this question would be processed through:

        1. **HyDE Enhancement**: Generating a hypothetical answer to improve retrieval
        2. **ColBERT Retrieval**: Finding relevant regulatory articles using advanced embeddings
        3. **Winnowing Filter**: Selecting the most relevant content
        4. **Answer Generation**: Creating a comprehensive response with citations

        The system would search through the Capital Requirements Regulation (CRR) and other regulatory documents to provide accurate, cited answers to your regulatory questions.
        """
        citations = [
            "Demo Mode - Article references would appear here",
            "Demo Mode - Multiple regulatory sources would be cited"
        ]
    
    return RAGResult(answer=answer.strip(), retrieved_articles=citations)

def process_question(question: str, k_value: int, system_ready: bool) -> RAGResult:
    """Wrapper to run async function in sync context."""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_question_async(question, k_value, system_ready))
        loop.close()
        return result
    except Exception as e:
        return RAGResult(
            answer=f"An error occurred: {str(e)}",
            retrieved_articles=[]
        )

def display_answer(result: RAGResult):
    """Display the answer and citations."""
    if not result:
        return
    
    # Display the answer
    st.markdown("### 📝 Answer")
    st.markdown(
        f'<div class="answer-container">{result["answer"]}</div>',
        unsafe_allow_html=True
    )
    
    # Display citations if available
    if result["retrieved_articles"]:
        st.markdown("### 📚 Sources")
        st.markdown(f"*Based on {len(result['retrieved_articles'])} regulatory article(s):*")
        
        # Show citations in an expandable section
        with st.expander(f"View {len(result['retrieved_articles'])} source citations", expanded=False):
            for i, citation in enumerate(result["retrieved_articles"], 1):
                st.markdown(
                    f'<div class="citation-item">{i}. {citation}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.warning("No source citations available for this answer.")

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">📋 Regulation RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 1.1rem;'>"
        "Ask questions about regulatory documents using advanced ColBERT retrieval"
        "</p>",
        unsafe_allow_html=True
    )
    
    # Sidebar
    system_ready = display_system_status()
    k_value = display_configuration_panel()
    selected_example = display_example_questions()
    
    # Show demo notice if system not ready
    if not system_ready:
        st.info("🔧 **Demo Mode**: System dependencies not fully configured. The interface will work with mock responses to demonstrate functionality.")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Question input
        st.markdown("### 🤔 Ask Your Question")
        
        # Use selected example question if available
        default_question = selected_example if selected_example else ""
        
        question = st.text_area(
            "Enter your regulatory question:",
            value=default_question,
            height=100,
            placeholder="e.g., What are the capital requirements for credit risk?",
            help="Ask any question about regulatory requirements, calculations, or definitions."
        )
        
        # Submit button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            submit_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    with col2:
        # Info panel
        st.markdown("### ℹ️ How it works")
        st.markdown("""
        1. **HyDE Enhancement**: Generates hypothetical answer to improve retrieval
        2. **ColBERT Retrieval**: Finds relevant regulatory articles
        3. **Winnowing Filter**: Selects most relevant content
        4. **Answer Generation**: Creates comprehensive response with citations
        """)
    
    # Clear functionality
    if clear_button:
        st.session_state.selected_question = ""
        st.rerun()
    
    # Process question
    if submit_button and question.strip():

        with st.spinner("🔄 Processing your question... This may take a moment."):
            # Show progress steps
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Enhancing query with HyDE...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            status_text.text("📚 Retrieving relevant articles...")
            progress_bar.progress(50)
            time.sleep(0.5)
            
            status_text.text("🎯 Filtering most relevant content...")
            progress_bar.progress(75)
            time.sleep(0.5)
            
            status_text.text("✍️ Generating comprehensive answer...")
            progress_bar.progress(90)
            
            # Process the question
            result = process_question(question, k_value, system_ready)
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        # Display results
        display_answer(result)

    elif submit_button:
        st.warning("Please enter a question before submitting.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888; font-size: 0.9rem;'>"
        "Powered by ColBERT RAG • Capital Requirements Regulation (CRR) Database"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
