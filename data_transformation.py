import pandas as pd
import json
import random
from datetime import datetime


class AviationDataTransformer:
    def __init__(self):
        self.training_data = []

    def load_csv(self, file_path):
        """Load CSV with proper encoding"""
        return pd.read_csv(file_path, encoding='utf-8')

    def transform_aircraft_inventory(self, df):
        """Transform aircraft inventory to training examples"""
        for _, row in df.iterrows():
            # Basic info prompt
            self.training_data.append({
                "prompt": f"Tell me about the {row['manufacturer']} {row['model']}",
                "completion": f"The {row['manufacturer']} {row['model']} is a {row['category']} aircraft with a capacity of {row['max_passengers']} passengers. It offers a range of {row['range_km']}km with fuel efficiency of {row['fuel_efficiency_km_per_liter']} km/liter. The aircraft features {row['new_features']}"
            })

            # Feature-focused prompt
            self.training_data.append({
                "prompt": f"What are the new features of the {row['model']}?",
                "completion": f"The {row['model']} introduces {row['new_features']}. These innovations make it ideal for {row['category']} operations."
            })

            # Specification prompt
            self.training_data.append({
                "prompt": f"What are the specifications of {row['model']}?",
                "completion": f"Specifications: {row['max_passengers']} passengers, {row['range_km']}km range, {row['dimensions_cubic_m']} cubic meters, {row['num_propellers']} propellers, ${row['price_million_usd']}M price point."
            })

    def transform_marketing_posts(self, df, min_engagement=7.0):
        """Transform high-performing marketing posts"""
        # Filter by engagement score
        high_performing = df[df['engagement_score'] >= min_engagement]

        for _, row in high_performing.iterrows():
            # Direct post example
            self.training_data.append({
                "prompt": f"Write a {row['platform']} {row['post_type']} post about aviation",
                "completion": row['content']
            })

            # Platform-specific prompt
            self.training_data.append({
                "prompt": f"Create a {row['post_type']} for {row['platform']} targeting {row['target_audience']}",
                "completion": f"{row['title']}\n\n{row['content']}\n\n{row['hashtags']}"
            })

    def transform_aviation_terms(self, df):
        """Transform aviation terminology"""
        for _, row in df.iterrows():
            # Technical definition
            self.training_data.append({
                "prompt": f"What is {row['term']}?",
                "completion": row['definition']
            })

            # Layman explanation
            self.training_data.append({
                "prompt": f"Explain {row['term']} in simple terms",
                "completion": row['layman_explanation']
            })

            # Category-based prompt
            self.training_data.append({
                "prompt": f"Tell me about {row['category']} term: {row['term']}",
                "completion": f"{row['term']} is a {row['category']} term. {row['definition']} In simpler words: {row['layman_explanation']}"
            })

    def transform_marketing_terms(self, df):
        """Transform marketing terminology"""
        for _, row in df.iterrows():
            # Usage example
            self.training_data.append({
                "prompt": f"How do I use '{row['term']}' in aviation marketing?",
                "completion": f"{row['definition']}. {row['usage_context']}. Example: {row['example_phrase']}"
            })

            # Application prompt
            self.training_data.append({
                "prompt": f"Write a sentence using the marketing term '{row['term']}'",
                "completion": row['example_phrase']
            })

    def create_combined_prompts(self, aircraft_df, posts_df):
        """Create prompts combining aircraft data with marketing style"""
        # Sample 50 combinations
        for _ in range(50):
            aircraft = aircraft_df.sample(1).iloc[0]
            post_style = posts_df[posts_df['engagement_score'] >= 8.0].sample(1).iloc[0]

            self.training_data.append({
                "prompt": f"Write a {post_style['platform']} post about the {aircraft['model']}'s fuel efficiency",
                "completion": f"Introducing the {aircraft['manufacturer']} {aircraft['model']}! ✈️ With an impressive {aircraft['fuel_efficiency_km_per_liter']} km/liter fuel efficiency and {aircraft['range_km']}km range, this {aircraft['category']} aircraft is setting new standards for sustainable aviation. {aircraft['new_features']} #Aviation #Sustainability #Innovation"
            })

    def save_to_jsonl(self, output_file='train.jsonl'):
        """Save training data to JSONL format"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in self.training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Created {len(self.training_data)} training examples")
        print(f"Saved to {output_file}")
        return output_file

    def process_all(self, aircraft_file, marketing_posts_file,
                    aviation_terms_file, marketing_terms_file):
        """Main processing function"""
        print("Loading CSVs...")
        aircraft_df = self.load_csv(aircraft_file)
        posts_df = self.load_csv(marketing_posts_file)
        aviation_terms_df = self.load_csv(aviation_terms_file)
        marketing_terms_df = self.load_csv(marketing_terms_file)

        print("Transforming aircraft inventory...")
        self.transform_aircraft_inventory(aircraft_df)

        print("Transforming marketing posts...")
        self.transform_marketing_posts(posts_df)

        print("Transforming aviation terms...")
        self.transform_aviation_terms(aviation_terms_df)

        print("Transforming marketing terms...")
        self.transform_marketing_terms(marketing_terms_df)

        print("Creating combined prompts...")
        self.create_combined_prompts(aircraft_df, posts_df)

        # Shuffle for better training
        random.shuffle(self.training_data)

        return self.save_to_jsonl()


# Usage
if __name__ == "__main__":
    transformer = AviationDataTransformer()

    # Process all CSV files
    output_file = transformer.process_all(
        aircraft_file='raw_data/aircraft_inventory_1000.csv',
        marketing_posts_file='raw_data/marketing_posts_1000.csv',
        aviation_terms_file='raw_data/aviation_terms_1000.csv',
        marketing_terms_file='raw_data/marketing_terms_1000.csv'
    )

    # Upload to S3 for SageMaker
    # import boto3
    # s3 = boto3.client('s3')
    # s3.upload_file(output_file, 'your-bucket', 'llama-training/train.jsonl')