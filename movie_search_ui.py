import streamlit as st
import pandas as pd
import numpy as np
import pickle, csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Helper Function for Star Rating
# -----------------------------
def imdb_to_stars(rating):
    if rating < 2.0:
        return '⭐'
    elif 2.0 <= rating < 4.0:
        return '⭐⭐'
    elif 4.0 <= rating < 6.0:
        return '⭐⭐⭐'
    elif 6.0 <= rating < 8.0:
        return '⭐⭐⭐⭐'
    else:
        return '⭐⭐⭐⭐⭐'

# -----------------------------
# Load Dataset & Embeddings
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-large-en-v1.5")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset\\cleaned_movies.csv", on_bad_lines='skip', quoting=csv.QUOTE_ALL, low_memory=False)
    df['plot'] = df['plot'].fillna(df['summary']).fillna("")
    return df

@st.cache_data
def load_embeddings():
    with open("movie_plot_embeddings.pki", "rb") as f:
        return pickle.load(f)

# -----------------------------
# Main App Layout
# -----------------------------
st.set_page_config(page_title="🎬 Movie Plot Search", layout="centered")
st.title("🔍 Semantic Movie Plot Search")

# Load resources
model = load_model()
df = load_data()
plot_embeddings = load_embeddings()

titles = df['title'].tolist()
years = df['year'].tolist()
genres = df['genre'].tolist()
ratings = df['imdb_rating'].values

# -----------------------------
# Search Function
# -----------------------------
def search_movies(query, df, plot_embeddings, model, top_n=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, plot_embeddings)[0]
    indices = np.argsort(similarities)[-top_n:][::-1]

    results = []
    for i in indices:
        row = df.iloc[i]

        # Apply filters
        if selected_genre != "All" and selected_genre not in row['genre']:
            continue
        if not (year_range[0] <= row['year'] <= year_range[1]):
            continue
        if not (rating_range[0] <= row['imdb_rating'] <= rating_range[1]):
            continue

        stars = imdb_to_stars(row['imdb_rating'])

        results.append({
            "title": row['title'],
            "year": row['year'],
            "genre": row['genre'],
            "rating": row['imdb_rating'],
            "stars": stars,
            "similarity": float(similarities[i]),
            "plot": row['plot']
        })

        if len(results) >= top_n:
            break

    return results

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔍 Filter Results")

# Genre filter
genres_unique = sorted(df['genre'].str.split(",").explode().str.strip().unique())
selected_genre = st.sidebar.selectbox("Filter by Genre", ["All"] + list(genres_unique))

# Year range filter
min_year, max_year = int(df['year'].min()), int(df['year'].max())
year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

# IMDb rating filter
rating_range = st.sidebar.slider("IMDb Rating Range", 0.0, 10.0, (0.0, 10.0))

# Input box for query
query = st.text_input("Describe a movie plot (e.g., 'A hacker joins a resistance group fighting AI'):")


if query:
    with st.spinner("Searching..."):
        #results = search_movies(query, plot_embeddings, titles, years, genres, ratings)
        results = search_movies(query, df, plot_embeddings, model)

    st.markdown("---")
    st.subheader("Top Matches:")

    for r in results:
        st.markdown(f"""
        ### {r['title']} ({r['year']})
        - Genre: {r['genre']}
        - IMDb Rating: {r['rating']} {r['stars']}
        - Similarity Score: `{r['similarity']:.4f}`
        """)
    
        # Optional plot detail toggle
        if st.checkbox(f"Show plot details for '{r['title']}'", key=r['title']):
            idx = titles.index(r['title'])
            st.write(df.iloc[idx]['plot'])

    st.markdown("---")