from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd

# Load pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load preprocessed articles from CSV
preprocessed_articles = pd.read_csv("preprocessed_articles.csv")

# Generate embeddings
def generate_embeddings(articles_df):
    texts = articles_df['content'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

embeddings = generate_embeddings(preprocessed_articles)

# Save embeddings for reuse
np.save("article_embeddings.npy", embeddings)
