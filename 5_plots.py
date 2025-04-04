import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import os

# Create visualizations directory if it doesn't exist
VISUALIZATION_DIR = "visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
df = pd.read_csv("preprocessed_articles.csv")
embeddings = np.load("article_embeddings.npy")

def create_word_frequency_plot():
    """Create word frequency distribution plot"""
    # Combine all content
    all_words = ' '.join(df['content']).split()
    word_freq = Counter(all_words).most_common(20)
    
    words, counts = zip(*word_freq)
    
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'word_frequency.png'))
    plt.close()

def create_wordcloud():
    """Generate and save wordcloud"""
    text = ' '.join(df['content'])
    wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(text)
    
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Article Contents')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'wordcloud.png'))
    plt.close()

def create_article_length_distribution():
    """Create histogram of article lengths"""
    article_lengths = df['content'].str.split().str.len()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=article_lengths, bins=50)
    plt.title('Distribution of Article Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'article_length_distribution.png'))
    plt.close()

def create_embedding_visualization():
    """Create t-SNE visualization of article embeddings"""
    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create interactive scatter plot
    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hover_data=[df['title']],
        title='t-SNE Visualization of Article Embeddings'
    )
    fig.write_html(os.path.join(VISUALIZATION_DIR, 'embedding_visualization.html'))

def create_similarity_heatmap():
    """Create similarity heatmap for a subset of articles"""
    # Take first 20 articles for visualization
    n_articles = 20
    subset_embeddings = embeddings[:n_articles]
    
    # Calculate cosine similarity
    similarity_matrix = np.dot(subset_embeddings, subset_embeddings.T)
    norms = np.linalg.norm(subset_embeddings, axis=1)
    similarity_matrix /= np.outer(norms, norms)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        xticklabels=df['title'][:n_articles],
        yticklabels=df['title'][:n_articles],
        cmap='coolwarm'
    )
    plt.title('Article Similarity Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'similarity_heatmap.png'))
    plt.close()

def main():
    print("Generating visualizations...")
    
    print("1. Creating word frequency plot...")
    create_word_frequency_plot()
    
    print("2. Creating wordcloud...")
    create_wordcloud()
    
    print("3. Creating article length distribution...")
    create_article_length_distribution()
    
    print("4. Creating embedding visualization...")
    create_embedding_visualization()
    
    print("5. Creating similarity heatmap...")
    create_similarity_heatmap()
    
    print("\nAll visualizations have been generated!")
    print(f"\nVisualizations saved in {VISUALIZATION_DIR}/:")
    print("- word_frequency.png")
    print("- wordcloud.png")
    print("- article_length_distribution.png")
    print("- embedding_visualization.html")
    print("- similarity_heatmap.png")

if __name__ == "__main__":
    main()
