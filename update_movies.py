import pandas as pd
import os

CSV_FILE = "cleaned_movies.csv"

def load_data():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file '{CSV_FILE}' not found.")
    return pd.read_csv(CSV_FILE)

def save_data(df):
    df.to_csv(CSV_FILE, index=False)
    print("✅ Data saved.")

def add_movie():
    df = load_data()
    
    title = input("Enter movie title: ")
    stars = input("Enter stars (comma-separated): ")
    directors = input("Enter directors (comma-separated): ")
    year = int(input("Enter year: "))
    genre = input("Enter genres (comma-separated): ")
    runtime = int(input("Enter runtime (minutes): "))
    ratingCount = int(input("Enter rating count: "))
    plot = input("Enter plot summary: ")
    summary = input("Enter short summary: ")
    imdb_rating = float(input("Enter IMDb rating: "))
    source = input("Enter IMDb ID (e.g., tt0059646): ")

    new_row = {
        "title": title,
        "stars": f"[{stars}]",
        "directors": f"[{directors}]",
        "year": year,
        "genre": f"[{genre}]",
        "runtime": runtime,
        "ratingCount": ratingCount,
        "plot": plot,
        "summary": summary,
        "imdb_rating": imdb_rating,
        "source": source
    }

    df = df.append(new_row, ignore_index=True)
    save_data(df)
    print("🎬 New movie added!")

def update_movie():
    df = load_data()
    print("\nAvailable movies:")
    for idx, title in enumerate(df['title']):
        print(f"{idx}: {title}")
    
    choice = int(input("Enter index of movie to update: "))
    print("\nEnter new values (leave blank to keep current value):")

    for col in df.columns:
        new_val = input(f"{col} [{df.at[choice, col]}]: ")
        if new_val.strip() != "":
            df.at[choice, col] = new_val.strip()

    save_data(df)
    print("✅ Movie updated!")

if __name__ == "__main__":
    action = input("Do you want to (a)dd or (u)pdate a movie? ").lower()
    if action == 'a':
        add_movie()
    elif action == 'u':
        update_movie()
    else:
        print("Invalid option.")