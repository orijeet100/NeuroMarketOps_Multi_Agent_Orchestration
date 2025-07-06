import urllib
import webbrowser

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import boto3
import json
import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import string

# Download nltk data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


class DualModeRAG:
    def __init__(self, sagemaker_endpoint="marketing-llm", region="us-east-2"):
        self.endpoint_name = sagemaker_endpoint
        self.region = region
        self.sagemaker_client = boto3.client('sagemaker-runtime', region_name=region)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Separate indices for different modes
        self.inventory_index_path = "inventory_faiss.idx"
        self.inventory_metadata_path = "inventory_metadata.pkl"
        self.inventory_bm25_path = "inventory_bm25.pkl"
        self.marketing_index_path = "marketing_faiss.idx"
        self.marketing_metadata_path = "marketing_metadata.pkl"

        self.inventory_chunks = []
        self.marketing_chunks = []
        self.inventory_index = None
        self.marketing_index = None
        self.inventory_bm25 = None
        self.tokenized_corpus = []

    def create_inventory_chunks(self, df: pd.DataFrame) -> List[Dict]:
        """Create narrative chunks for inventory with contextual embeddings"""
        chunks = []

        for idx, row in df.iterrows():
            # Create narrative description
            narrative_parts = [
                f"Aircraft {row['aircraft_id']} is a {row['manufacturer']} {row['model']} {row['category']} aircraft.",
                f"It has a capacity of {row['max_passengers']} passengers and a range of {row['range_km']} kilometers.",
                f"The fuel efficiency is {row['fuel_efficiency_km_per_liter']} km/liter with {row['num_propellers']} propellers.",
                f"This aircraft has dimensions of {row['dimensions_cubic_m']} cubic meters and is painted in {row['primary_color']} colors.",
                f"Priced at ${row['price_million_usd']} million USD, it was introduced in {row['year_introduced']}.",
                f"Key features include: {row['new_features']}.",
                f"This aircraft was added to our inventory on {row['date_of_creation']}."
            ]

            narrative_text = " ".join(narrative_parts)

            # Add contextual information for better embeddings
            context_prompt = f"This is information about {row['manufacturer']} {row['model']}, a {row['category']} aircraft used for {'long-haul' if row['range_km'] > 10000 else 'medium-haul' if row['range_km'] > 5000 else 'short-haul'} flights with {'excellent' if row['fuel_efficiency_km_per_liter'] > 4.5 else 'good' if row['fuel_efficiency_km_per_liter'] > 3.5 else 'standard'} fuel efficiency."

            chunk_text = f"{context_prompt} {narrative_text}"

            chunks.append({
                "text": chunk_text,
                "aircraft_id": row['aircraft_id'],
                "model": row['model'],
                "manufacturer": row['manufacturer'],
                "date": row['date_of_creation']
            })

        return chunks

    def create_marketing_chunks(self, dfs: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Create chunks for marketing content generation"""
        chunks = []

        # Marketing posts
        if 'marketing_posts' in dfs:
            df = dfs['marketing_posts']
            for _, row in df.iterrows():
                chunk_text = f"Platform: {row['platform']}\nType: {row['post_type']}\nTitle: {row['title']}\nContent: {row['content']}\nEngagement: {row['engagement_score']}"
                chunks.append({"text": chunk_text, "type": "post", "platform": row['platform']})

        # Marketing terms only
        if 'marketing_terms' in dfs:
            df = dfs['marketing_terms']
            for _, row in df.iterrows():
                chunk_text = f"Term: {row['term']}\nDefinition: {row['definition']}"
                chunks.append({"text": chunk_text, "type": "term"})

        return chunks

    def tokenize_text(self, text):
        """Simple tokenizer without NLTK dependency"""
        text = text.lower()
        # Remove punctuation
        for char in string.punctuation:
            text = text.replace(char, ' ')
        # Split on whitespace and filter empty strings
        return [word for word in text.split() if word]

    def load_or_create_indices(self):
        """Load or create separate indices including BM25"""
        # Inventory index
        if os.path.exists(self.inventory_index_path) and os.path.exists(self.inventory_bm25_path):
            try:
                self.inventory_index = faiss.read_index(self.inventory_index_path)
                with open(self.inventory_metadata_path, 'rb') as f:
                    self.inventory_chunks = pickle.load(f)
                with open(self.inventory_bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                    self.inventory_bm25 = bm25_data['bm25']
                    self.tokenized_corpus = bm25_data['corpus']
            except Exception as e:
                st.error(f"Error loading inventory indices: {str(e)}")
                self.create_inventory_index()
        else:
            self.create_inventory_index()

        # Marketing index
        if os.path.exists(self.marketing_index_path):
            try:
                self.marketing_index = faiss.read_index(self.marketing_index_path)
                with open(self.marketing_metadata_path, 'rb') as f:
                    self.marketing_chunks = pickle.load(f)
            except Exception as e:
                st.error(f"Error loading marketing index: {str(e)}")
                self.create_marketing_index()
        else:
            self.create_marketing_index()

    def create_inventory_index(self):
        """Create index for inventory only with BM25"""
        path = 'raw_data/aircraft_inventory_1000_unique.csv'
        if not os.path.exists(path):
            st.error(f"Aircraft inventory file not found at {path}")
            self.inventory_chunks = []
            self.inventory_index = faiss.IndexFlatL2(384)
            self.inventory_bm25 = None
            return

        try:
            df = pd.read_csv(path)
            if df.empty:
                st.error("Aircraft inventory file is empty")
                return

            st.info(f"Loading aircraft inventory: {len(df)} rows")
            self.inventory_chunks = self.create_inventory_chunks(df)

            texts = [chunk['text'] for chunk in self.inventory_chunks]

            # Create FAISS index
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
            self.inventory_index = faiss.IndexFlatL2(embeddings.shape[1])
            self.inventory_index.add(embeddings)

            # Create BM25 index
            self.tokenized_corpus = [self.tokenize_text(text) for text in texts]
            self.inventory_bm25 = BM25Okapi(self.tokenized_corpus)

            # Save all indices
            faiss.write_index(self.inventory_index, self.inventory_index_path)
            with open(self.inventory_metadata_path, 'wb') as f:
                pickle.dump(self.inventory_chunks, f)
            with open(self.inventory_bm25_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.inventory_bm25,
                    'corpus': self.tokenized_corpus
                }, f)
        except Exception as e:
            st.error(f"Error loading aircraft inventory: {str(e)}")
            print(f"Detailed error: {e}")
            import traceback
            traceback.print_exc()
            self.inventory_chunks = []
            self.inventory_index = faiss.IndexFlatL2(384)
            self.inventory_bm25 = None

    def create_marketing_index(self):
        """Create index for marketing content"""
        dfs = {}
        for file, name in [
            ('marketing_posts_1000.csv', 'marketing_posts'),
            ('marketing_terms_1000.csv', 'marketing_terms')
        ]:
            path = f'raw_data/{file}'
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if not df.empty:
                        dfs[name] = df
                        st.info(f"Loaded {name}: {len(df)} rows")
                except pd.errors.EmptyDataError:
                    st.warning(f"File {file} is empty")
                except Exception as e:
                    st.error(f"Error loading {file}: {str(e)}")
            else:
                st.warning(f"File not found: {path}")

        if not dfs:
            st.error("No marketing data found! Creating empty index.")
            self.marketing_chunks = []
            self.marketing_index = faiss.IndexFlatL2(384)  # Default embedding size
            return

        self.marketing_chunks = self.create_marketing_chunks(dfs)

        texts = [chunk['text'] for chunk in self.marketing_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        self.marketing_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.marketing_index.add(embeddings)

        faiss.write_index(self.marketing_index, self.marketing_index_path)
        with open(self.marketing_metadata_path, 'wb') as f:
            pickle.dump(self.marketing_chunks, f)

    def hybrid_search_inventory(self, query: str, n_results: int = 50, alpha: float = 0.5) -> List[Dict]:
        """Hybrid search combining FAISS and BM25 with adjustable weight"""
        # FAISS search
        query_embedding = self.embedder.encode([query])
        faiss_distances, faiss_indices = self.inventory_index.search(query_embedding,
                                                                     min(n_results, len(self.inventory_chunks)))

        # Normalize FAISS scores (inverse of distance)
        faiss_scores = 1 / (1 + faiss_distances[0])
        faiss_scores = faiss_scores / np.max(faiss_scores) if np.max(faiss_scores) > 0 else faiss_scores

        # BM25 search
        tokenized_query = self.tokenize_text(query)
        bm25_scores = self.inventory_bm25.get_scores(tokenized_query)
        bm25_scores = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores

        # Combine scores
        results = []
        seen_ids = set()

        # Process FAISS results
        for idx, (faiss_idx, faiss_score) in enumerate(zip(faiss_indices[0], faiss_scores)):
            if faiss_idx < len(self.inventory_chunks):
                chunk = self.inventory_chunks[faiss_idx].copy()
                chunk['faiss_score'] = float(faiss_score)
                chunk['bm25_score'] = float(bm25_scores[faiss_idx])
                chunk['hybrid_score'] = alpha * chunk['faiss_score'] + (1 - alpha) * chunk['bm25_score']
                chunk['faiss_distance'] = float(faiss_distances[0][idx])
                results.append(chunk)
                seen_ids.add(faiss_idx)

        # Add high BM25 scoring items not in FAISS results
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:n_results]
        for idx in bm25_top_indices:
            if idx not in seen_ids and idx < len(self.inventory_chunks):
                chunk = self.inventory_chunks[idx].copy()
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

    def search_marketing(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search marketing content"""
        query_embedding = self.embedder.encode([query])
        distances, indices = self.marketing_index.search(query_embedding, n_results)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.marketing_chunks):
                chunk = self.marketing_chunks[idx].copy()
                chunk['relevance_score'] = 1 / (1 + distance)
                results.append(chunk)
        return results


    def query_sagemaker(self, prompt, message_placeholder):
        """Stream directly to Streamlit"""
        # payload = {
        #     "inputs": prompt,
        #     "parameters": {
        #         "max_new_tokens": 256,
        #         "temperature": 0.6,
        #         "top_p": 0.9,
        #         "stream": True
        #     }
        # }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.5,
                "top_p": 0.85,
                "top_k": 50,
                "repetition_penalty": 1.3,
                "do_sample": False,
                "stream": True,
        "stop": ["<|eot_id|>", "<|end_of_text|>"]  # Proper stop tokens
            }
        }

        try:
            response = self.sagemaker_client.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            full_text = ""
            for event in response['Body']:
                chunk = event.get('PayloadPart', {}).get('Bytes', b'').decode('utf-8')
                if chunk:
                    try:
                        data = json.loads(chunk)
                        if 'token' in data and 'text' in data['token']:
                            full_text += data['token']['text']
                            message_placeholder.markdown(full_text + "▌")
                    except:
                        pass

            message_placeholder.markdown(full_text)
            return full_text

        except Exception as e:
            return f"Error: {e}"


# Streamlit App
st.set_page_config(page_title="Aviation AI Assistant", layout="wide")

# Initialize system
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = DualModeRAG()
    st.session_state.rag_system.load_or_create_indices()

if 'mode' not in st.session_state:
    st.session_state.mode = 'inventory'

if 'marketing_messages' not in st.session_state:
    st.session_state.marketing_messages = []

if 'inventory_messages' not in st.session_state:
    st.session_state.inventory_messages = []

# Mode selector
col1, col2 = st.columns(2)
with col1:
    if st.button("✍️ Marketing Agent", use_container_width=True,
                 type="primary" if st.session_state.mode == 'marketing' else "secondary"):
        st.session_state.mode = 'marketing'
with col2:
    if st.button("📊 RAG-powered Manager", use_container_width=True,
                 type="primary" if st.session_state.mode == 'inventory' else "secondary"):
        st.session_state.mode = 'inventory'

# Display appropriate interface
if st.session_state.mode == 'marketing':
    st.title("✍️ Marketing Agent")
    st.caption("Specializes in post and content generation for LinkedIn, Twitter, etc.")

    # Chat history with context
    for message in st.session_state.marketing_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    col1, col2 = st.columns([10, 1])
    with col2:
        has_content = len(st.session_state.marketing_messages) > 0 and st.session_state.marketing_messages[-1][
            'role'] == 'assistant'

        if st.button("🔗",
                     disabled=not has_content,
                     help="Post to LinkedIn" if has_content else "Generate content first"):
            if has_content:
                content = st.session_state.marketing_messages[-1]['content']
                linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?text={urllib.parse.quote(content[:1300])}"
                webbrowser.open(linkedin_url)
    if prompt := st.chat_input("Request marketing content..."):
        st.session_state.marketing_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Build context with history (no RAG for marketing)
            history = ""
            if len(st.session_state.marketing_messages) > 1:
                recent = st.session_state.marketing_messages[-6:-1]
                history = "\n".join([f"{m['role']}: {m['content']}..." for m in recent]) + "\n\n"

            full_prompt = f'''
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are a helpful and creative Aviation Marketing Assistant. Your job is to generate accurate and engaging, and brand-aligned detailed content or responses based on the user’s prompt and prior conversation history.
            
            Please keep your response:
            - Clear
            - Relevant to aviation marketing
            - Focused only on the task asked
            - Avoid unnecessary elaboration, disclaimers, or repetition
            
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Here is the history of the previous conversation (if any): 
            {history}

            Now, based on this, answer the following query or complete the task:
            {prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            '''

            print(history)
            message_placeholder = st.empty()
            full_response = st.session_state.rag_system.query_sagemaker(full_prompt, message_placeholder)

            st.session_state.marketing_messages.append({"role": "assistant", "content": full_response})

else:  # Inventory mode
    st.title("📊 RAG-powered Manager")
    st.caption("Precise information from aircraft inventory database")

    # Retrieval settings in sidebar
    with st.sidebar:
        st.header("Retrieval Settings")

        # Hybrid search weight
        st.subheader("Hybrid Search")
        alpha = st.slider("Semantic vs Keyword Weight",
                          min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                          help="0 = Pure BM25 (keyword), 1 = Pure FAISS (semantic)")

        retrieval_mode = st.radio("Retrieval Mode", ["Top K", "Distance Threshold"])

        if retrieval_mode == "Top K":
            top_k = st.slider("Number of results (Top K)", min_value=1, max_value=20, value=10)
            threshold = None
        else:
            threshold = st.slider("Hybrid Score Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            top_k = 50  # Get many results for filtering

    # No history for inventory queries
    for message in st.session_state.inventory_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask about aircraft inventory..."):
        st.session_state.inventory_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # Hybrid search inventory
            all_results = st.session_state.rag_system.hybrid_search_inventory(query, n_results=top_k, alpha=alpha)

            # Filter based on mode
            if retrieval_mode == "Top K":
                inventory_results = all_results[:top_k]
            else:
                inventory_results = [r for r in all_results if r['hybrid_score'] >= threshold]

            # Display results in a table
            if inventory_results:
                # st.info(
                #     f"Found {len(inventory_results)} aircraft (α={alpha}: {int(alpha * 100)}% semantic, {int((1 - alpha) * 100)}% keyword)")
                #
                # results_data = []
                # for i, result in enumerate(inventory_results):
                #     results_data.append({
                #         "Rank": i + 1,
                #         "Aircraft": f"{result['manufacturer']} {result['model']}",
                #         "Hybrid Score": f"{result['hybrid_score']:.3f}",
                #         "FAISS": f"{result['faiss_score']:.3f}",
                #         "BM25": f"{result['bm25_score']:.3f}",
                #         "Details": result['row_summary']
                #     })
                #
                # df = pd.DataFrame(results_data)
                # st.dataframe(df, use_container_width=True, height=400)

                # Use all filtered results for context
                # context = "\n\n---\n\n".join([r['text'] for r in inventory_results])
                context = "\n\n".join([r['text'] for r in inventory_results])
                full_prompt = prompt = f"""
                
                
                <|start_header_id|>user<|end_header_id|>
You are an aircraft inventory assistant who is precise and friendly. Based on the following inventory data, generate a succinct and structured response.

Inventory Data:
{context}

Question: {query}

Answer:
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
                
                
                """
            else:
                st.warning("No results found with current settings")
                full_prompt = f"No aircraft found matching the query: {query}. Please explain that no results were found."

            # Debug: Show what's being sent
            with st.expander("Debug: Prompt sent to model", expanded=False):
                st.text(full_prompt + "...")

            message_placeholder = st.empty()
            full_response = st.session_state.rag_system.query_sagemaker(full_prompt, message_placeholder)
            st.session_state.inventory_messages.append({"role": "assistant", "content": full_response})


# Sidebar
with st.sidebar:
    if st.session_state.mode == 'marketing':
        st.header("Marketing Agent")
        st.info("Generates content using examples from marketing posts, terms, and best practices")
        if st.button("Clear Chat"):
            st.session_state.marketing_messages = []
            st.rerun()
    else:
        st.header("Inventory Manager")
        st.info("Searches only aircraft inventory for precise specifications and details")
        if st.button("Clear Results"):
            st.session_state.inventory_messages = []
            st.rerun()
