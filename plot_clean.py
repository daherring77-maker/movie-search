import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from nltk.tokenize import sent_tokenize
import nltk

#nltk.download('punkt')     # For general sentence tokenization
#nltk.download('punkt_tab') # For tab-aware tokenization (your case)

# Load the model (lightweight, fast, accurate)
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_split_plot(plot_text, threshold=1.2):
    """
    Splits a long plot summary into semantically coherent paragraphs.
    
    Args:
        plot_text (str): Long plot text
        threshold (float): Cluster similarity threshold
    
    Returns:
        list: List of cleaned paragraphs
    """
    if not isinstance(plot_text, str) or len(plot_text.strip()) == 0:
        return [""]

    # Tokenize into sentences
    sentences = sent_tokenize(plot_text)

    # If only one sentence, return it directly
    if len(sentences) <= 1:
        return sentences

    # Generate embeddings
    embeddings = model.encode(sentences)

    # Cluster similar sentences
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
    cluster_assignment = clustering_model.fit_predict(embeddings)

    # Group by clusters
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(sentences[sentence_id])

    # Sort and return
    sorted_paragraphs = [" ".join(clustered_sentences[k]) for k in sorted(clustered_sentences)]

    return sorted_paragraphs


def clean_plots_in_csv(input_file="cleaned_movies.csv", output_file="cleaned_final.csv"):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    if "plot" not in df.columns:
        raise ValueError("Input CSV must contain a 'plot' column")

    print("Splitting plot summaries into semantic paragraphs...")
    df["plot"] = df["plot"].apply(
        lambda x: "\n\n".join(semantic_split_plot(x))
    )

    print(f"Saving updated data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("✅ Done! File saved.")


if __name__ == "__main__":
    clean_plots_in_csv()