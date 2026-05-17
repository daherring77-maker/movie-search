import pandas as pd
from titlecase import titlecase

def clean_movie_titles(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Ensure the 'title' column exists
    if 'title' not in df.columns:
        raise ValueError("Input CSV must contain a 'title' column")
    
    # Apply title casing to the 'title' column
    df['title'] = df['title'].astype(str).apply(titlecase)
    
    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Cleaned titles saved to {output_file}")

if __name__ == "__main__":
    input_file = "movie_plots.csv"  # Replace with your input CSV file name
    output_file = "cleaned_titles.csv"  # Output file name
    
    clean_movie_titles(input_file, output_file)