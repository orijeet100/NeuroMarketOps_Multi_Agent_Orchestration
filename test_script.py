import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import boto3
import json
import os
from rank_bm25 import BM25Okapi
import string


class InventoryRAG:
    def __init__(self, endpoint="marketing-llm", region="us-east-2"):
        self.endpoint = endpoint
        self.sagemaker = boto3.client('sagemaker-runtime', region_name=region)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Load indices
        self.load_indices()

    def load_indices(self):
        """Load pre-built indices"""
        try:
            self.index = faiss.read_index("inventory_faiss.idx")
            with open("inventory_metadata.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
            with open("inventory_bm25.pkl", 'rb') as f:
                bm25_data = pickle.load(f)
                self.bm25 = bm25_data['bm25']
                self.corpus = bm25_data['corpus']
            print(f"Loaded {len(self.chunks)} inventory chunks")
        except Exception as e:
            print(f"Error loading indices: {e}")
            exit(1)

    def tokenize(self, text):
        text = text.lower()
        for char in string.punctuation:
            text = text.replace(char, ' ')
        return [word for word in text.split() if word]

    def search(self, query, k=5, alpha=0.5):
        """Hybrid search"""
        print(f"\nSearching for: '{query}'")
        print(f"Using alpha={alpha} (semantic vs keyword weight)")

        # FAISS search
        query_emb = self.embedder.encode([query])
        distances, indices = self.index.search(query_emb, k)

        # BM25 search
        tokenized_query = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores
        faiss_scores = 1 / (1 + distances[0])
        faiss_scores = faiss_scores / np.max(faiss_scores) if np.max(faiss_scores) > 0 else faiss_scores
        bm25_scores = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores

        # Combine and sort
        results = []
        for idx, faiss_score in zip(indices[0], faiss_scores):
            hybrid_score = alpha * faiss_score + (1 - alpha) * bm25_scores[idx]
            results.append({
                'chunk': self.chunks[idx],
                'score': hybrid_score
            })

        results.sort(key=lambda x: x['score'], reverse=True)

        print(f"\nTop {k} results:")
        for i, r in enumerate(results[:k]):
            print(f"{i + 1}. {r['chunk']['manufacturer']} {r['chunk']['model']} (score: {r['score']:.3f})")

        return results[:k]

    def query_sagemaker(self, prompt):
        """Query endpoint with streaming"""
        print("\n" + "=" * 50)
        print("Sending to SageMaker...")

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": True
            }
        }

        try:
            response = self.sagemaker.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint,
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            print("\nResponse: ", end='', flush=True)
            full_text = ""

            for event in response['Body']:
                chunk = event.get('PayloadPart', {}).get('Bytes', b'').decode('utf-8')
                if chunk:
                    try:
                        data = json.loads(chunk)
                        if 'token' in data and 'text' in data['token']:
                            text = data['token']['text']
                            print(text, end='', flush=True)
                            full_text += text
                    except:
                        pass

            print()  # New line at end
            return full_text

        except Exception as e:
            print(f"Error: {e}")
            return None

    def answer_question(self, question):
        """Full RAG pipeline"""
        # Search inventory
        results = self.search(question, k=5, alpha=0.5)

        # Build context
        context = "\n\n".join([r['chunk']['text'] for r in results])

        # Create prompt
        prompt = f"""You are an aircraft inventory assistant. Based on the following inventory data, answer the question.

Inventory Data:
{context}

Question: {question}

Answer:"""

        # Get response
        response = self.query_sagemaker(prompt)

        if response:
            print("\n" + "=" * 50)
            print("ANSWER:")
            print("=" * 50)
            print(response)
        else:
            print("\nNo response generated")


# Main
if __name__ == "__main__":
    print("Initializing RAG system...")
    rag = InventoryRAG()

    print("\n" + "=" * 50)
    print("AIRCRAFT INVENTORY Q&A")
    print("=" * 50)

    while True:
        question = input("\nYour question (or 'quit'): ")
        if question.lower() == 'quit':
            break

        rag.answer_question(question)