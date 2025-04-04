import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the saved embeddings
embeddings = np.load("article_embeddings.npy")
dimension = embeddings.shape[1]

# Load preprocessed articles
preprocessed_articles = pd.read_csv("preprocessed_articles.csv")

# Load FAISS
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
index.add(embeddings)

# Function to recommend articles
def recommend_articles(titles, k=10):
    # Get contents of articles with given titles
    query_articles = preprocessed_articles[preprocessed_articles['title'].isin(titles)]
    if query_articles.empty:
        print("No articles found with given titles")
        return []
    
    query_contents = query_articles['content'].tolist()
    query_embedding = model.encode(query_contents)
    
    # If multiple query articles, take average of their embeddings
    if len(query_contents) > 1:
        query_embedding = np.mean(query_embedding, axis=0, keepdims=True)
    
    distances, indices = index.search(query_embedding, k + len(titles))  # Get extra results to account for excluding query articles
    
    # Create a set of indices to exclude (corresponding to query articles)
    exclude_indices = set(preprocessed_articles[preprocessed_articles['title'].isin(titles)].index)
    
    # Filter out the query articles from recommendations
    filtered_recommendations = []
    for i, d in zip(indices[0], distances[0]):
        if i not in exclude_indices and len(filtered_recommendations) < k:
            filtered_recommendations.append({
                "title": preprocessed_articles.iloc[i]['title'],
                "score": 1 - d
            })
    
    return filtered_recommendations

# Example usage
query_titles = ["165th meridian east", "155th meridian east"]  # List of titles to find similar articles for
recommendations = recommend_articles(query_titles)

for rec in recommendations:
    print(f"Title: {rec['title']}, Score: {rec['score']:.4f}")
