import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "wiki_movie_plots_deduped.csv"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Change to "qwen3-embedding-4b" if available
EMBEDDING_CACHE_FILE = "movie_plot_embeddings.pkl"

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
plots = df['Plot'].tolist()
titles = df['Title'].tolist()
years = df['Release Year'].tolist()
genres = df['Genre'].tolist()

print(f"Loaded {len(plots)} movie plots.")

# -----------------------------
# Setup Device and Model
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
model = SentenceTransformer(EMBEDDING_MODEL_NAME).to(device)

# -----------------------------
# Encode Plots or Load Cached Embeddings
# -----------------------------
if os.path.exists(EMBEDDING_CACHE_FILE):
    print("Loading cached embeddings...")
    with open(EMBEDDING_CACHE_FILE, 'rb') as f:
        plot_embeddings = pickle.load(f)
else:
    print("Encoding movie plots (this may take a few minutes)...")
    plot_embeddings = model.encode(plots, batch_size=64, show_progress_bar=True)
    print("Caching embeddings for future use...")
    with open(EMBEDDING_CACHE_FILE, 'wb') as f:
        pickle.dump(plot_embeddings, f)

# -----------------------------
# Semantic Search Function
# -----------------------------
def search_movies(query, top_n=5):
    print("\nProcessing query:", query)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(query_embedding, plot_embeddings)[0]
    indices = np.argsort(similarities)[-top_n:][::-1]

    print(f"\nTop {top_n} matching movies:\n")
    results = []
    for i in indices:
        title = titles[i]
        year = years[i]
        genre = genres[i]
        score = similarities[i]
        print(f"{title} ({year}, {genre}) - Similarity: {score:.4f}")
        results.append({
            "title": title,
            "year": year,
            "genre": genre,
            "similarity": float(score)
        })
    return results

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a movie description (or type 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        search_movies(user_input, top_n=5)
