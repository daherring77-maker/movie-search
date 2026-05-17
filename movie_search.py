import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os, csv

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "dataset\\cleaned_final.csv"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5" 
EMBEDDING_CACHE_FILE = "movie_plot_embeddings.pki" 
""

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")
#df = pd.read_csv("movie_plots.csv", on_bad_lines='skip', low_memory=False)
df = pd.read_csv("dataset\\cleaned_movies.csv", on_bad_lines='skip', quoting=csv.QUOTE_ALL, low_memory=False)

# Fill missing 'plot' using 'summary'
df['plot'] = df['plot'].fillna(df['summary'])

# Then fill any remaining NaNs with empty string
df['plot'] = df['plot'].fillna("")

# Convert to list of strings
plots = [str(plot) for plot in df['plot']]


plots = df['plot'].tolist()
titles = df['title'].tolist()
years = df['year'].tolist()
genres = df['genre'].tolist()
ratings = df['imdb_rating'].values

#stars_rating = df['stars_rating'].tolist()
print(df['plot'].isnull().sum())
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
   # plot_embeddings = model.encode(plots, batch_size=64, query_chunk_size=4, show_progress_bar=True)
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
        imdb_rating = ratings[i]
        score = similarities[i]
        
        #stars_rating = stars_rating[i]
        print(f"{title} ({year}, {genre}, {imdb_rating}) - Similarity: {score:.4f}")
        results.append({
            "title": title,
            "year": year,
            "genre": genre,
            "imdb_rating" : float(imdb_rating),
            "similarity": float(score),

           # "stars": stars_rating
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
