import streamlit as st
import os
import sys
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool, tool
from typing import Type, List, Dict
from pydantic import BaseModel, Field
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
import string
import time
from datetime import datetime
import re

# Set encoding to UTF-8 to handle special characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set page config
st.set_page_config(page_title="Aviation AI Assistant", layout="wide", page_icon="✈️")

# Initialize OpenAI
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
    if api_key:
        st.session_state.openai_api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.info("""
    **Workflow Steps:**
    1. 🔀 Router Agent analyzes query
    2. 📦 Inventory, 📝 Marketing, or 🤝 General path
    3. 🤖 Specialized agent processes request
    4. ✅ Response delivered
    """)

    st.markdown("""
    **Agent Types:**
    - 📦 **Inventory**: Aircraft searches & specs
    - 📝 **Marketing**: Content creation
    - 🤝 **General**: Guidance & friendly chat
    """)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.context_history = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_history" not in st.session_state:
    st.session_state.context_history = []
if "crew_system" not in st.session_state:
    st.session_state.crew_system = None


# Tools and Agents
class InventorySearchInput(BaseModel):
    query: str = Field(..., description="Search query for aircraft inventory")


class InventorySearchTool(BaseTool):
    name: str = "search_aircraft_inventory"
    description: str = "Search our aircraft inventory CSV database for specifications, prices, and aircraft details"
    args_schema: Type[BaseModel] = InventorySearchInput

    def _run(self, query: str) -> str:
        """Execute hybrid search combining FAISS and BM25 with improved logic"""
        try:
            # Clean query to handle encoding issues
            query = self._clean_text(query)

            # Load resources
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            index = faiss.read_index("inventory_faiss.idx")

            with open("inventory_metadata.pkl", 'rb') as f:
                chunks = pickle.load(f)

            with open("inventory_bm25.pkl", 'rb') as f:
                bm25_data = pickle.load(f)
                bm25 = bm25_data['bm25']

            # Perform hybrid search
            results = self._hybrid_search_logic(query, embedder, index, chunks, bm25)

            if not results:
                return "NO_RESULTS_FOUND"

            # Format results for display with encoding safety
            formatted_results = []
            for result in results[:5]:  # Top 10 results
                # Clean text content to avoid encoding issues
                manufacturer = self._clean_text(result.get('manufacturer', ''))
                model = self._clean_text(result.get('model', ''))
                text = self._clean_text(result.get('text', ''))

                formatted_results.append(
                    f"**{manufacturer} {model}** "
                    f"(Hybrid Score: {result['hybrid_score']:.3f})\n"
                    f"{text}"
                )

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            return f"Error searching inventory: {error_msg}"

    def _clean_text(self, text: str) -> str:
        """Clean text to handle encoding issues"""
        if not isinstance(text, str):
            text = str(text)

        # Remove or replace problematic characters
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _hybrid_search_logic(self, query: str, embedder, index, chunks, bm25,
                             n_results: int = 50, alpha: float = 0.3):
        """Hybrid search combining FAISS and BM25 with adjustable weight"""

        # FAISS search
        query_embedding = embedder.encode([query])
        faiss_distances, faiss_indices = index.search(query_embedding,
                                                      min(n_results, len(chunks)))

        # Normalize FAISS scores (inverse of distance)
        faiss_scores = 1 / (1 + faiss_distances[0])
        faiss_scores = faiss_scores / np.max(faiss_scores) if np.max(faiss_scores) > 0 else faiss_scores

        # BM25 search
        tokenized_query = self._tokenize_text(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_scores = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores

        # Combine scores
        results = []
        seen_ids = set()

        # Process FAISS results
        for idx, (faiss_idx, faiss_score) in enumerate(zip(faiss_indices[0], faiss_scores)):
            if faiss_idx < len(chunks):
                chunk = chunks[faiss_idx].copy()
                chunk['faiss_score'] = float(faiss_score)
                chunk['bm25_score'] = float(bm25_scores[faiss_idx])
                chunk['hybrid_score'] = alpha * chunk['faiss_score'] + (1 - alpha) * chunk['bm25_score']
                chunk['faiss_distance'] = float(faiss_distances[0][idx])
                results.append(chunk)
                seen_ids.add(faiss_idx)

        # Add high BM25 scoring items not in FAISS results
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:n_results]
        for idx in bm25_top_indices:
            if idx not in seen_ids and idx < len(chunks):
                chunk = chunks[idx].copy()
                chunk['faiss_score'] = 0.0
                chunk['bm25_score'] = float(bm25_scores[idx])
                chunk['hybrid_score'] = (1 - alpha) * chunk['bm25_score']
                chunk['faiss_distance'] = 999.0  # High distance for non-FAISS results
                results.append(chunk)

        # Sort by hybrid score
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        # Extract key details from narrative
        for result in results:
            text_lines = result['text'].split('.')
            for line in text_lines:
                if 'is a' in line and result['manufacturer'] in line:
                    result['row_summary'] = line.strip()
                    break
            else:
                result['row_summary'] = f"{result['manufacturer']} {result['model']}"

        return results[:n_results]

    def _tokenize_text(self, text: str):
        """Simple tokenization for BM25"""
        import re
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()


@tool("search_web")
def search_web(query: str) -> str:
    """Search the web for current aviation information"""
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(f"aviation aircraft {query} 2024")
        return results[:1500]
    except Exception as e:
        return f"Web search error: {str(e)}"


@tool("general_web_search")
def general_web_search(query: str) -> str:
    """Search the web for general information"""
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return results[:1000]  # Shorter for general queries
    except Exception as e:
        return f"Web search error: {str(e)}"


# Enhanced Aviation Crew System
class StreamingAviationCrew:
    def __init__(self, llm):
        self.llm = llm
        self.setup_agents()

    def setup_agents(self):
        # Router Agent with enhanced classification
        self.router_agent = Agent(
            role='Intelligent Request Router',
            goal='Analyze requests and classify them into the most appropriate category',
            backstory="""You are an intelligent router that analyzes user requests and conversation context.
            You classify queries into three categories:
            - INVENTORY: Aircraft searches, specifications, pricing, availability
            - MARKETING: Content creation, marketing materials, promotional content
            - GENERAL: Greetings, general questions, help requests, casual conversation, guidance

            Consider the conversation history to provide accurate classification.""",
            llm=self.llm,
            verbose=True
        )

        # Inventory Manager
        self.inventory_manager = Agent(
            role='Aircraft Inventory Specialist',
            goal='Search inventory and provide detailed aircraft information',
            backstory="""You are an expert in aircraft inventory management. You search our database
            for aircraft specifications, pricing, and availability. When items aren't found,
            you offer to search online resources for the user.""",
            tools=[InventorySearchTool()],
            llm=self.llm,
            verbose=True
        )

        # Marketing Agent
        self.marketing_agent = Agent(
            role='Aviation Marketing Specialist',
            goal='Create compelling marketing content for aviation industry',
            backstory="""You are a creative marketing expert specializing in aviation content.
            You create engaging marketing materials, promotional content, and can reference
            previous conversations to maintain context and continuity.""",
            tools=[search_web],
            llm=self.llm,
            verbose=True
        )

        # General Purpose Agent
        self.general_agent = Agent(
            role='Friendly Aviation Assistant',
            goal='Provide helpful guidance and friendly conversation for general queries',
            backstory="""You are a friendly and knowledgeable aviation assistant who helps with
            general questions, provides guidance, and engages in casual conversation. You have
            broad knowledge about aviation and can help users understand how to use this system.
            You're warm, approachable, and always ready to help users feel comfortable.""",
            tools=[general_web_search],
            llm=self.llm,
            verbose=True
        )

    def process_with_streaming(self, user_query: str, context_history: List[Dict], progress_container):
        """Process request with step-by-step visibility"""

        # Step 1: Enhanced Routing
        with progress_container.container():
            st.info("🔀 **Step 1:** Analyzing query and routing to appropriate agent...")

            # Add context to routing decision
            context_str = ""
            if context_history:
                recent_context = context_history[-3:]  # Last 3 exchanges
                context_str = "Recent conversation context:\n"
                for ctx in recent_context:
                    context_str += f"{ctx['role']}: {ctx['content'][:100]}...\n"

            routing_task = Task(
                description=f"""
                {context_str}

                Current User Query: "{user_query}"

                Analyze this query and classify it into one of these categories:

                🔍 INVENTORY: 
                - Aircraft searches (e.g., "Find Boeing 737", "Show me Cessna models")
                - Specifications requests (e.g., "What are the specs of...", "Price of...")
                - Availability queries (e.g., "Do you have...", "Is X aircraft available")

                📝 MARKETING:
                - Content creation (e.g., "Write a description", "Create marketing copy")
                - Promotional materials (e.g., "Make an ad", "Write a brochure")
                - Marketing strategy (e.g., "How to promote...", "Marketing ideas")

                🤝 GENERAL:
                - Greetings (e.g., "Hello", "Hi there", "Good morning")
                - General questions (e.g., "How does this work?", "What can you do?")
                - Help requests (e.g., "Help me", "I need guidance", "Explain...")
                - Casual conversation (e.g., "How are you?", "Tell me about aviation")
                - System guidance (e.g., "What features do you have?")

                Output ONLY the classification word: INVENTORY, MARKETING, or GENERAL
                """,
                agent=self.router_agent,
                expected_output="Single word classification: INVENTORY, MARKETING, or GENERAL"
            )

            routing_crew = Crew(
                agents=[self.router_agent],
                tasks=[routing_task],
                process=Process.sequential,
                verbose=True
            )

            route_result = routing_crew.kickoff()

            # Extract decision
            if hasattr(route_result, 'result'):
                route_decision = route_result.result.strip().upper()
            else:
                route_decision = str(route_result).strip().upper()

            # Ensure we have a valid route
            if route_decision not in ['INVENTORY', 'MARKETING', 'GENERAL']:
                route_decision = 'GENERAL'  # Default fallback

            st.success(f"✅ Router Decision: **{route_decision}**")
            time.sleep(0.5)

        # Step 2: Process based on route with loading bar
        with progress_container.container():
            if "INVENTORY" in route_decision:
                st.info("📦 **Step 2:** Forwarding to Inventory Specialist Agent...")

                # Add loading bar for inventory processing
                loading_text = "🔍 Searching aircraft inventory database..."
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text(loading_text)

                # Simulate loading progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("🔍 Searching aircraft inventory database...")
                    elif i < 60:
                        status_text.text("📊 Processing aircraft specifications...")
                    elif i < 90:
                        status_text.text("💰 Checking pricing and availability...")
                    else:
                        status_text.text("✅ Compiling results...")

                task = Task(
                    description=f"""
                    User Query: {user_query}

                    Search our aircraft inventory for the requested information. The user query may be unstructured, 
                    but you should find the closest matches and provide detailed information.

                    When you receive search results from the inventory tool :
                    1. If you get formatted aircraft results (with manufacturer, model, hybrid scores, etc.), 
                       analyze and see which ones matches closely
                    2. Highlight the most relevant match based on the user's query
                    3. Give accurate and concise answers based on the query 

                    ONLY if the tool returns exactly "NO_RESULTS_FOUND" should you respond with:
                    "I couldn't find that specific aircraft in our current inventory. 
                    Would you like me to search online resources for general information about it?"

                    Otherwise, present the found aircraft information in a clear, organized manner.
                    """,
                    agent=self.inventory_manager,
                    expected_output="Detailed inventory information with specifications and pricing, or offer to search online if no results"
                )

                crew = Crew(
                    agents=[self.inventory_manager],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )

            elif "MARKETING" in route_decision:
                st.info("📝 **Step 2:** Forwarding to Marketing Specialist Agent...")

                # Add loading bar for marketing processing
                loading_text = "✏️ Creating marketing content..."
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text(loading_text)

                # Simulate loading progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 25:
                        status_text.text("✏️ Analyzing marketing requirements...")
                    elif i < 50:
                        status_text.text("🎯 Crafting compelling content...")
                    elif i < 75:
                        status_text.text("🔍 Researching aviation trends...")
                    else:
                        status_text.text("✅ Finalizing marketing materials...")

                # Include relevant context for marketing
                marketing_context = ""
                if context_history:
                    # Find relevant previous discussions
                    for ctx in context_history[-5:]:  # Last 5 exchanges
                        if any(word in ctx.get('content', '').lower() for word in
                               ['aircraft', 'aviation', 'plane', 'jet']):
                            marketing_context += f"Previous context: {ctx['content'][:200]}...\n"

                task = Task(
                    description=f"""
                    {marketing_context}

                    Marketing Request: {user_query}

                    Create compelling marketing content based on this request. 
                    Use any relevant context from previous conversations to maintain continuity.
                    Make the content engaging, professional, and aviation-focused.
                    """,
                    agent=self.marketing_agent,
                    expected_output="Professional marketing content"
                )

                crew = Crew(
                    agents=[self.marketing_agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )

            else:  # GENERAL
                st.info("🤝 **Step 2:** Forwarding to General Assistant Agent...")

                # Add loading bar for general processing
                loading_text = "🤖 Processing your request..."
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text(loading_text)

                # Simulate loading progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 40:
                        status_text.text("🤖 Understanding your question...")
                    elif i < 80:
                        status_text.text("💭 Preparing helpful response...")
                    else:
                        status_text.text("✅ Finalizing answer...")

                # Include conversation context for general queries
                general_context = ""
                if context_history:
                    recent_context = context_history[-2:]  # Last 2 exchanges
                    general_context = "Recent conversation:\n"
                    for ctx in recent_context:
                        general_context += f"{ctx['role']}: {ctx['content'][:150]}...\n"

                task = Task(
                    description=f"""
                    {general_context}

                    User Query: {user_query}

                    Respond in a friendly, helpful manner. This could be:
                    - A greeting or casual conversation
                    - A general question about aviation or this system
                    - A request for help or guidance
                    - A general inquiry that doesn't fit inventory or marketing

                    Be warm, approachable, and informative. If the user seems confused about 
                    what this system can do, explain the three main capabilities:
                    1. 📦 Aircraft inventory search and specifications
                    2. 📝 Marketing content creation for aviation
                    3. 🤝 General assistance and friendly conversation

                    If you need current information to answer properly, use the web search tool.
                    """,
                    agent=self.general_agent,
                    expected_output="Friendly, helpful response appropriate to the query type"
                )

                crew = Crew(
                    agents=[self.general_agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )

            # Execute the task
            result = crew.kickoff()

            # Clear the loading elements
            progress_bar.empty()
            status_text.empty()

            # Extract final result
            if hasattr(result, 'result'):
                final_output = result.result
            else:
                final_output = str(result)

            st.success("✅ **Step 3:** Processing complete!")

            return final_output, route_decision


# Main UI
st.title("✈️ Aviation AI Assistant")
st.caption("Intelligent multi-agent system with inventory search, marketing content creation, and general assistance")

# Feature overview
with st.expander("🚀 What can I help you with?", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📦 Aircraft Inventory**
        - Search aircraft specifications
        - Check availability and pricing
        - Compare different models
        - Get detailed technical info
        """)

    with col2:
        st.markdown("""
        **📝 Marketing Content**
        - Create marketing materials
        - Write product descriptions
        - Generate promotional content
        - Develop marketing strategies
        """)

    with col3:
        st.markdown("""
        **🤝 General Assistance**
        - Answer aviation questions
        - Provide system guidance
        - Friendly conversation
        - Help with general inquiries
        """)

# Check API key
if not st.session_state.openai_api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start.")
    st.stop()

# Initialize crew system
if st.session_state.crew_system is None:
    with st.spinner("Initializing multi-agent AI system..."):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        st.session_state.crew_system = StreamingAviationCrew(llm)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "system":
            st.info(message["content"])
        else:
            st.markdown(message["content"])

# Chat input with examples
if prompt := st.chat_input("Ask about aircraft inventory, request marketing content, or just say hello..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with step visibility
    with st.chat_message("assistant"):
        progress_container = st.empty()

        try:
            # Process request
            response, route_type = st.session_state.crew_system.process_with_streaming(
                prompt,
                st.session_state.context_history,
                progress_container
            )

            # Clear progress and show response
            progress_container.empty()

            # Format response based on agent type
            if route_type == "INVENTORY":
                if "NO_RESULTS_FOUND" in response or "couldn't find" in response.lower():
                    st.warning(response)
                    if st.button("🌐 Search Online Instead"):
                        st.info("Searching online resources...")
                else:
                    st.markdown(response)

            elif route_type == "MARKETING":
                st.markdown(response)

            else:  # GENERAL
                st.markdown(response)

            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.context_history.append({
                "role": "user",
                "content": prompt,
                "route": route_type,
                "timestamp": datetime.now()
            })
            st.session_state.context_history.append({
                "role": "assistant",
                "content": response,
                "route": route_type,
                "timestamp": datetime.now()
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("💡 Try rephrasing your question or check your API key configuration.")

# Footer
st.divider()
st.caption(
    "Powered by CrewAI with OpenAI GPT-3.5 | Multi-Agent Architecture: Router → Inventory/Marketing/General → Response")