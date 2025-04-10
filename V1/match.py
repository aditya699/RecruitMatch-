"""
CV-Job Matcher

Compares extracted CV data to job descriptions and calculates match scores
based on relevance of qualifications, experience, and skills.

Author: Aditya Bhatt
"""

import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import csv

def load_data(cv_file="cv_data.csv", job_file="job_summaries.csv"):
    """
    Load CV and job data from CSV files
    
    Args:
        cv_file (str): Path to CV data CSV
        job_file (str): Path to job summaries CSV
        
    Returns:
        tuple: (cv_data, job_data) DataFrames
    """
    try:
        cv_data = pd.read_csv(cv_file)
        job_data = pd.read_csv(job_file)
        
        print(f"Loaded {len(cv_data)} CVs and {len(job_data)} job descriptions")
        return cv_data, job_data
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def create_composite_text(row, is_cv=True):
    """
    Create a composite text representation of CV or job data
    
    Args:
        row (Series): Row from DataFrame
        is_cv (bool): Whether row is CV data (True) or job data (False)
        
    Returns:
        str: Composite text representation
    """
    if is_cv:
        # For CV data
        composite = f"Name: {row['Name']}. "
        
        if not pd.isna(row['Skills']):
            composite += f"Skills: {row['Skills']}. "
            
        if not pd.isna(row['Education']):
            composite += f"Education: {row['Education']}. "
            
        if not pd.isna(row['Experience']):
            composite += f"Experience: {row['Experience']}. "
            
        if not pd.isna(row['Certifications']) and row['Certifications']:
            composite += f"Certifications: {row['Certifications']}. "
        
        return composite
    else:
        # For job data
        composite = f"Job Title: {row['Job Title']}. "
        
        if not pd.isna(row['Skills']):
            composite += f"Skills: {row['Skills']}. "
            
        if not pd.isna(row['Experience']):
            composite += f"Experience: {row['Experience']}. "
            
        if not pd.isna(row['Qualifications']):
            composite += f"Qualifications: {row['Qualifications']}. "
            
        if not pd.isna(row['Responsibilities']):
            composite += f"Responsibilities: {row['Responsibilities']}. "
        
        return composite

def get_embeddings(text, model="llama3.1"):
    """
    Get embeddings for text using Ollama API
    
    Args:
        text (str): Text to embed
        model (str): Model to use for embeddings
        
    Returns:
        list: Embedding vector
    """
    url = "http://localhost:11434/api/embeddings"
    
    data = {
        "model": model,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            print(f"Error getting embeddings: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception getting embeddings: {str(e)}")
        return None

def process_data_for_matching(cv_data, job_data):
    """
    Process CV and job data to create embeddings
    
    Args:
        cv_data (DataFrame): CV data
        job_data (DataFrame): Job data
        
    Returns:
        tuple: (cv_embeddings, job_embeddings, cv_texts, job_texts)
    """
    # Create composite texts
    cv_texts = cv_data.apply(lambda row: create_composite_text(row, is_cv=True), axis=1)
    job_texts = job_data.apply(lambda row: create_composite_text(row, is_cv=False), axis=1)
    
    # Get embeddings
    print("Getting embeddings for CVs...")
    cv_embeddings = []
    for i, text in enumerate(cv_texts):
        print(f"  Processing CV {i+1}/{len(cv_texts)}")
        embedding = get_embeddings(text)
        if embedding:
            cv_embeddings.append(embedding)
        else:
            # Use zeros vector as fallback
            cv_embeddings.append([0] * 3072)  # llama3 embedding size
    
    print("Getting embeddings for jobs...")
    job_embeddings = []
    for i, text in enumerate(job_texts):
        print(f"  Processing job {i+1}/{len(job_texts)}")
        embedding = get_embeddings(text)
        if embedding:
            job_embeddings.append(embedding)
        else:
            # Use zeros vector as fallback
            job_embeddings.append([0] * 3072)  # llama3 embedding size
    
    return np.array(cv_embeddings), np.array(job_embeddings), cv_texts, job_texts

def calculate_matches(cv_data, job_data, cv_embeddings, job_embeddings):
    """
    Calculate match scores between CVs and jobs
    
    Args:
        cv_data (DataFrame): CV data
        job_data (DataFrame): Job data
        cv_embeddings (ndarray): CV embeddings
        job_embeddings (ndarray): Job embeddings
        
    Returns:
        list: Match results
    """
    results = []
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(cv_embeddings, job_embeddings)
    
    # For each job, find top 5 CV matches
    for job_idx in range(len(job_data)):
        job_title = job_data.iloc[job_idx]['Job Title']
        
        # Get scores for this job
        job_scores = similarity_matrix[:, job_idx]
        
        # Get indices of top 5 matches
        top_indices = np.argsort(job_scores)[-5:][::-1]
        
        job_matches = []
        for rank, cv_idx in enumerate(top_indices):
            match_score = job_scores[cv_idx]
            cv_name = cv_data.iloc[cv_idx]['Name']
            cv_filename = cv_data.iloc[cv_idx]['Filename']
            
            job_matches.append({
                'rank': rank + 1,
                'cv_name': cv_name,
                'cv_filename': cv_filename,
                'match_score': match_score
            })
        
        results.append({
            'job_title': job_title,
            'matches': job_matches
        })
    
    return results

def save_match_results(results, output_file="match_results.csv", db_file="match_results.db"):
    """
    Save match results to CSV file and SQLite database
    
    Args:
        results (list): Match results
        output_file (str): Output CSV file path
        db_file (str): Output SQLite DB file path
        
    Returns:
        None
    """
    # Save to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Job Title', 'Rank', 'Candidate Name', 'CV Filename', 'Match Score'])
        
        # Write data
        for job_result in results:
            job_title = job_result['job_title']
            
            for match in job_result['matches']:
                writer.writerow([
                    job_title,
                    match['rank'],
                    match['cv_name'],
                    match['cv_filename'],
                    f"{match['match_score']:.4f}"
                ])
    
    print(f"Match results saved to {output_file}")
    
    # Save to SQLite
    import sqlite3
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS match_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_title TEXT NOT NULL,
        rank INTEGER NOT NULL,
        candidate_name TEXT NOT NULL,
        cv_filename TEXT NOT NULL,
        match_score REAL NOT NULL
    )
    ''')
    
    # Insert data
    for job_result in results:
        job_title = job_result['job_title']
        
        for match in job_result['matches']:
            cursor.execute(
                "INSERT INTO match_results (job_title, rank, candidate_name, cv_filename, match_score) VALUES (?, ?, ?, ?, ?)",
                (
                    job_title,
                    match['rank'],
                    match['cv_name'],
                    match['cv_filename'],
                    match['match_score']
                )
            )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Match results saved to SQLite database: {db_file}")

def main():
    """Main entry point for the script"""
    # Load data
    cv_data, job_data = load_data()
    if cv_data is None or job_data is None:
        return
    
    # Process data for matching
    cv_embeddings, job_embeddings, cv_texts, job_texts = process_data_for_matching(cv_data, job_data)
    
    # Calculate matches
    results = calculate_matches(cv_data, job_data, cv_embeddings, job_embeddings)
    
    # Save results
    save_match_results(results)

if __name__ == "__main__":
    main()