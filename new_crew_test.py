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
from langchain.callbacks.base import BaseCallbackHandler

# --- Streamlit Streaming Callback ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

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
    api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password", key="api_key_input")
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
    stream_placeholder = st.empty()
    handler = StreamlitCallbackHandler(stream_placeholder)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True, callbacks=[handler])
    from crew_ai_test import StreamingAviationCrew  # Adjust import path
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        handler = StreamlitCallbackHandler(stream_placeholder)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True, callbacks=[handler])
        st.session_state.crew_system.llm = llm

        try:
            response, route_type = st.session_state.crew_system.process_with_streaming(
                prompt,
                st.session_state.context_history,
                stream_placeholder
            )

            stream_placeholder.empty()

            if route_type == "INVENTORY":
                if "NO_RESULTS_FOUND" in response or "couldn't find" in response.lower():
                    st.warning(response)
                    if st.button("🌐 Search Online Instead"):
                        st.info("Searching online resources...")
                else:
                    st.markdown(response)

            elif route_type == "MARKETING":
                st.markdown(response)

            else:
                st.markdown(response)

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
