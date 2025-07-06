import pandas as pd
import numpy as np
from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import boto3
import json
import os
import sys


class CSVRAGSystem:
    def __init__(self, sagemaker_endpoint="marketing-llm", region="us-east-1"):
        """Initialize RAG system with FAISS"""
        self.endpoint_name = sagemaker_endpoint
        self.region = region
        self.sagemaker_client = boto3.client('sagemaker-runtime', region_name=region)

        # Initialize embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Storage paths
        self.index_path = "faiss_index.idx"
        self.metadata_path = "metadata.pkl"
        self.chunks = []
        self.index = None

    def create_row_chunks(self, df: pd.DataFrame, table_name: str, chunk_size: int = 5) -> List[Dict]:
        """Convert CSV rows into contextual chunks with rich metadata"""
        chunks = []

        table_descriptions = {
            "aircraft_inventory": "Aircraft specifications and features database"
            # "marketing_posts": "Marketing content across platforms",
            # "aviation_terms": "Technical aviation terminology",
            # "marketing_terms": "Marketing terminology and usage"
        }

        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]

            context_parts = [
                f"Source: {table_descriptions.get(table_name, table_name)}",
                f"Table: {table_name}",
                f"Data Type: {self.get_data_type(table_name)}"
            ]

            if 'date_of_creation' in df.columns:
                dates = chunk_df['date_of_creation'].dropna()
                if len(dates) > 0:
                    context_parts.append(f"Date Range: {dates.min()} to {dates.max()}")

            context_parts.append("---")

            for idx, row in chunk_df.iterrows():
                row_parts = [f"Record #{idx}:"]

                if table_name == "aircraft_inventory" and 'model' in row:
                    row_parts.insert(0, f"[Aircraft: {row.get('manufacturer', '')} {row.get('model', '')}]")
                elif table_name == "marketing_posts" and 'platform' in row:
                    row_parts.insert(0,
                                     f"[{row.get('platform', '')} Post - Score: {row.get('engagement_score', 'N/A')}]")

                for col, val in row.items():
                    if pd.notna(val):
                        row_parts.append(f"{col}: {val}")

                context_parts.append(" | ".join(row_parts))

            chunk_text = "\n".join(context_parts)

            chunks.append({
                "text": chunk_text,
                "table": table_name,
                "row_ids": list(chunk_df.index),
                "metadata": {
                    "table": table_name,
                    "source_type": self.get_data_type(table_name),
                    "row_count": len(chunk_df),
                    "columns": list(df.columns),
                    "date_range": self.get_date_range(chunk_df) if 'date_of_creation' in df.columns else None
                }
            })

        return chunks

    def get_data_type(self, table_name: str) -> str:
        """Get descriptive data type for table"""
        types = {
            "aircraft_inventory": "Technical Specifications"
            # "marketing_posts": "Marketing Content",
            # "aviation_terms": "Technical Dictionary",
            # "marketing_terms": "Marketing Dictionary"
        }
        return types.get(table_name, "General Data")

    def get_date_range(self, df: pd.DataFrame) -> str:
        """Get date range from dataframe"""
        if 'date_of_creation' in df.columns:
            dates = pd.to_datetime(df['date_of_creation'], errors='coerce').dropna()
            if len(dates) > 0:
                return f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
        return None

    def load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors")
        else:
            print("Creating new index...")
            self.load_all_csvs()
            self.create_faiss_index()

    def load_all_csvs(self):
        """Load all CSVs from raw_data folder"""
        csv_files = {
            'aircraft_inventory_1000.csv': 'aircraft_inventory'
            # 'aviation_terms_1000.csv': 'aviation_terms',
            # 'aviation_terms_1000.csv': 'aviation_terms',
            # 'marketing_posts_1000.csv': 'marketing_posts',
            # 'marketing_terms_1000.csv': 'marketing_terms'
        }

        raw_data_path = "raw_data"

        for csv_file, table_name in csv_files.items():
            file_path = os.path.join(raw_data_path, csv_file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Loading {table_name}: {len(df)} rows")

                chunks = self.create_row_chunks(df, table_name)
                self.chunks.extend(chunks)
                print(f"Created {len(chunks)} chunks from {table_name}")

    def create_faiss_index(self):
        """Create FAISS index from chunks"""
        if not self.chunks:
            print("No chunks to index!")
            return

        # Extract texts and create embeddings
        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Save index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.chunks, f)

        print(f"Created index with {self.index.ntotal} vectors")

    def search_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant chunks using FAISS"""
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding, n_results)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk['relevance_score'] = 1 / (1 + distance)
                results.append(chunk)

        return results

    def query_sagemaker_endpoint(self, prompt: str) -> str:
        """Query SageMaker with streaming output"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

        response = self.sagemaker_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        result = json.loads(response['Body'].read().decode())
        generated_text = result[0]['generated_text'] if isinstance(result, list) else result

        # Stream output
        for char in generated_text:
            sys.stdout.write(char)
            sys.stdout.flush()

        return generated_text

    def generate_response(self, query: str, n_context: int = 5) -> str:
        """Generate response using RAG"""
        relevant_chunks = self.search_relevant_chunks(query, n_context)

        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"From {chunk['table']}:\n{chunk['text']}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are an aviation marketing expert. Use the following context to answer the question.
Pay attention to the source type and dates of the information provided.

Context:
{context}

Question: {query}

Provide a detailed answer based on the context above. If the context mentions specific dates, aircraft models, or engagement scores, include them in your response.

Answer: """

        print("\nGenerating response...\n")
        response = self.query_sagemaker_endpoint(prompt)
        print("\n")

        return response


# Usage
if __name__ == "__main__":
    rag = CSVRAGSystem(sagemaker_endpoint="marketing-llm", region="us-east-2")

    # Load or create index
    rag.load_or_create_index()

    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Type 'exit' to quit")
    print("=" * 60)

    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break

        # Show retrieved context
        chunks = rag.search_relevant_chunks(query, n_results=3)
        print("\nRetrieved Context:")
        for i, chunk in enumerate(chunks):
            print(f"\n{i + 1}. From {chunk['table']} (relevance: {chunk['relevance_score']:.3f})")
            print(chunk['text'][:200] + "...")

        response = rag.generate_response(query)
        print(f"\nFull response saved.",response)