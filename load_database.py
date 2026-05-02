import pandas as pd
import sqlite3

print("--- Starting Database Loading Script ---")

# --- 1. Configuration ---
CSV_FILE_PATH = 'dataset_withsa.csv'
DB_FILE_PATH = 'hospital_reviews.db'

# --- 2. Create Database Connection ---
# This command creates the database file if it doesn't exist.
conn = sqlite3.connect(DB_FILE_PATH)
cursor = conn.cursor()
print(f"Successfully connected to database at '{DB_FILE_PATH}'")

# --- 3. Define and Create Database Schema (Tables) ---

print("Creating tables if they don't exist...")

print("Creating 'users' table if it doesn't exist...")
# Table for user authentication
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);
""")

# Table for unique hospitals
cursor.execute("""
CREATE TABLE IF NOT EXISTS hospitals (
    hospital_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hospital_name TEXT UNIQUE NOT NULL
);
""")

# Table for unique authors
cursor.execute("""
CREATE TABLE IF NOT EXISTS authors (
    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
    author_name TEXT UNIQUE NOT NULL
);
""")

# The main table for all reviews, linking to the others
cursor.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hospital_id INTEGER,
    author_id INTEGER,
    star_rating INTEGER NOT NULL,
    review_text_raw TEXT,
    processed_content TEXT,
    sentiment_score REAL,
    predicted_sentiment TEXT,
    FOREIGN KEY (hospital_id) REFERENCES hospitals (hospital_id),
    FOREIGN KEY (author_id) REFERENCES authors (author_id)
);
""")
print("Tables created successfully.")

# --- 4. Load CSV Data using Pandas ---
print(f"Loading data from '{CSV_FILE_PATH}'...")
df = pd.read_csv(CSV_FILE_PATH)
print("CSV data loaded successfully.")

# --- 5. Populate the 'hospitals' and 'authors' Tables ---
# Get unique values from the specified columns and insert them.
# "OR IGNORE" gracefully handles any duplicates.
unique_hospitals = df['Hospital_Name'].dropna().unique()
cursor.executemany("INSERT OR IGNORE INTO hospitals (hospital_name) VALUES (?)", [(name,) for name in unique_hospitals])

unique_authors = df['Author name'].dropna().unique()
cursor.executemany("INSERT OR IGNORE INTO authors (author_name) VALUES (?)", [(name,) for name in unique_authors])

conn.commit()
print("Populated 'hospitals' and 'authors' tables with unique values.")

# --- 6. Prepare and Populate the 'reviews' Table ---
# Get the IDs we just created for the hospitals and authors to link them in the reviews table
hospital_map = {name: id for name, id in cursor.execute("SELECT hospital_name, hospital_id FROM hospitals")}
author_map = {name: id for name, id in cursor.execute("SELECT author_name, author_id FROM authors")}

# Map the names in the original DataFrame to their corresponding new IDs
df['hospital_id'] = df['Hospital_Name'].map(hospital_map)
df['author_id'] = df['Author name'].map(author_map)

# Select the final columns needed for the reviews table
reviews_to_insert = df[[
    'hospital_id',
    'author_id',
    'Star rating',
    'Review content',      # The raw review text
    'processed_content',
    'Sentiment Score',     # The numeric score
    'Predicted Sentiment'
]]

# Rename the DataFrame columns to match our clean database schema names exactly
reviews_to_insert = reviews_to_insert.rename(columns={
    'Star rating': 'star_rating',
    'Review content': 'review_text_raw',
    'Sentiment Score': 'sentiment_score',
    'Predicted Sentiment': 'predicted_sentiment'
})

# Use pandas' to_sql() function for efficient bulk insertion into the 'reviews' table
print("Populating 'reviews' table... This may take a moment.")
reviews_to_insert.to_sql('reviews', conn, if_exists='append', index=False)
print("Successfully populated the 'reviews' table.")

# --- 7. Finalize ---
conn.commit()
conn.close()

print("\n--- Database loading complete! ---")
print(f"Your data has been successfully loaded into '{DB_FILE_PATH}'.")