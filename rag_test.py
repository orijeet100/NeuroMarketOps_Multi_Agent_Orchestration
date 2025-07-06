import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os


class TestCSVRetrieval:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./test_chroma_db")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collections = {}

    def load_and_index_csv(self, csv_path, table_name):
        """Load CSV and create simple row-based embeddings"""
        df = pd.read_csv(csv_path)
        print(f"\nLoading {table_name}: {len(df)} rows")

        # Create collection
        collection = self.chroma_client.get_or_create_collection(
            name=table_name,
            embedding_function=self.embedding_function
        )
        self.collections[table_name] = collection

        # Index each row as a document
        documents = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            # Create searchable text from row
            text_parts = []

            # Add key identifiers first
            if table_name == "aircraft_inventory":
                text_parts.append(f"Aircraft: {row.get('manufacturer', '')} {row.get('model', '')}")
            elif table_name == "marketing_posts":
                text_parts.append(f"{row.get('platform', '')} post")

            # Add all column values
            for col, val in row.items():
                if pd.notna(val) and str(val).strip():
                    text_parts.append(f"{col}: {str(val)}")

            doc_text = " | ".join(text_parts)
            documents.append(doc_text)

            # Metadata
            metadata = {
                "table": table_name,
                "row_index": idx
            }

            # Add specific important fields to metadata
            if 'date_of_creation' in row:
                metadata['date'] = str(row['date_of_creation'])
            if 'engagement_score' in row:
                metadata['engagement'] = float(row['engagement_score'])
            if 'category' in row:
                metadata['category'] = str(row['category'])

            metadatas.append(metadata)
            ids.append(f"{table_name}_row_{idx}")

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )

        print(f"Indexed {len(documents)} rows from {table_name}")

    def test_retrieval(self, query, n_results=5):
        """Test retrieval for a query"""
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print('=' * 60)

        for table_name, collection in self.collections.items():
            print(f"\nSearching in {table_name}...")

            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count())
            )

            if results['documents'][0]:
                print(f"Top {len(results['documents'][0])} results:")
                for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                )):
                    print(f"\n{i + 1}. Distance: {distance:.4f}")
                    print(f"   Metadata: {metadata}")
                    print(f"   Content: {doc[:200]}...")
            else:
                print("No results found")

    def run_tests(self):
        """Run comprehensive retrieval tests"""
        # Load all CSVs
        csv_files = {
            'raw_data/aircraft_inventory_1000.csv': 'aircraft_inventory',
            'raw_data/aviation_terms_1000.csv': 'aviation_terms',
            'raw_data/marketing_posts_1000.csv': 'marketing_posts',
            'raw_data/marketing_terms_1000.csv': 'marketing_terms'
        }

        for csv_file, table_name in csv_files.items():
            if os.path.exists(csv_file):
                self.load_and_index_csv(csv_file, table_name)

        # Test queries
        test_queries = [
            "Corporate branding"
        ]

        for query in test_queries:
            self.test_retrieval(query, n_results=3)

        # Specific pattern tests
        print(f"\n{'=' * 60}")
        print("PATTERN TEST: Content Search")
        print('=' * 60)

        # Direct content search
        for collection_name, collection in self.collections.items():
            if collection_name == "marketing_posts":
                # Search for specific content
                results = collection.query(
                    query_texts=["composite materials carbon fiber"],
                    n_results=5,
                    where={"table": "marketing_posts"}
                )
                print(f"\nDirect search in {collection_name}:")
                print(f"Found {len(results['documents'][0])} results")
                for doc in results['documents'][0][:2]:
                    if 'composite' in doc.lower() or 'carbon' in doc.lower():
                        print(f"✓ Found relevant content: {doc[:150]}...")
                    else:
                        print(f"✗ Irrelevant result: {doc[:150]}...")


if __name__ == "__main__":
    tester = TestCSVRetrieval()
    tester.run_tests()