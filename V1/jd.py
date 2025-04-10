"""

Job Description Summarizer: Reads and summarizes key elements from the JD, including required skills, experience, qualifications, and job responsibilities.

Reads and summarizes key elements from job descriptions including:
- Required skills
- Experience
- Qualifications 
- Job responsibilities

Author: Aditya Bhatt
"""

import pandas as pd
import sqlite3
from ollama import chat
from pydantic import BaseModel
from typing import List, Optional

class JobSummary(BaseModel):
    """Data model for structured job summary information"""
    skills: List[str]
    experience: str
    qualifications: List[str]
    responsibilities: List[str]

class JDReader:
    """
    Reads job descriptions from CSV files and extracts structured information
    using LLM-powered analysis
    """
    
    def __init__(self, file_path):
        """
        Initialize with path to job description CSV file
        
        Args:
            file_path (str): Path to CSV file with job descriptions
        """
        self.file_path = file_path
       
    def analyze_csv(self):
        """
        Read CSV file with appropriate encoding and extract job data
        
        Returns:
            DataFrame: Pandas DataFrame with job titles and descriptions
            None: If reading fails
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    # Read CSV file into pandas DataFrame with different encoding
                    data = pd.read_csv(self.file_path, encoding=encoding)
                    data = data[['Job Title', 'Job Description']]
                        
                    # Print first few rows
                    print("\nSample data (first 5 rows):")
                    print(data.head())
                    
                    return data
                except UnicodeDecodeError:
                    continue
                    
            print("Failed to read with any encoding")
            return None
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None
    
    def summarize_jobs(self, model='llama3.1'):
        """
        Process job descriptions with LLM to extract structured information
        
        Args:
            model (str): Ollama model name to use
            
        Returns:
            list: List of job summary dictionaries
            None: If no data available
        """
        data = self.analyze_csv()
        if data is None:
            return None
            
        results = []
        total_jobs = len(data)
        
        for index, row in data.iterrows():
            job_title = row['Job Title']
            job_desc = row['Job Description']
            
            print(f"Processing job {index+1}/{total_jobs}: {job_title}")
            
            # Create prompt for the LLM
            prompt = f"""
            Based on the following job description, extract key elements:
            
            JOB TITLE: {job_title}
            JOB DESCRIPTION: {job_desc}
            
            Extract skills, experience requirements, qualifications, and responsibilities.
            """
            
            # Get structured response from Ollama
            try:
                response = chat(
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                        }
                    ],
                    model=model,
                    format=JobSummary.model_json_schema(),
                )
                
                # Parse the response into our Pydantic model
                summary = JobSummary.model_validate_json(response.message.content)
                
                # Add to results with job title
                results.append({
                    'job_title': job_title,
                    'summary': summary
                })
                
            except Exception as e:
                print(f"Error processing job '{job_title}': {str(e)}")
                
        return results

def save_to_sqlite(summaries, db_file="job_summaries.db"):
    """
    Save extracted job summaries to SQLite database in a single table
    
    Args:
        summaries (list): List of job summary dictionaries
        db_file (str): Path to SQLite database file
        
    Returns:
        None
    """
    if not summaries:
        print("No summaries to save")
        return
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create a single table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS job_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_title TEXT NOT NULL,
        skills TEXT,
        experience TEXT,
        qualifications TEXT,
        responsibilities TEXT
    )
    ''')
    
    # Insert data
    for job in summaries:
        cursor.execute(
            "INSERT INTO job_summaries (job_title, skills, experience, qualifications, responsibilities) VALUES (?, ?, ?, ?, ?)",
            (
                job['job_title'], 
                ', '.join(job['summary'].skills),
                job['summary'].experience,
                ', '.join(job['summary'].qualifications),
                ', '.join(job['summary'].responsibilities)
            )
        )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Summaries saved to SQLite database: {db_file}")

def display_summaries(summaries):
    """
    Print job summaries to console in formatted output
    
    Args:
        summaries (list): List of job summary dictionaries
        
    Returns:
        None
    """
    for job in summaries:
        print(f"\n--- {job['job_title']} ---")
        summary = job['summary']
        
        print("Skills:")
        for skill in summary.skills:
            print(f"- {skill}")
            
        print(f"\nExperience: {summary.experience}")
        
        print("\nQualifications:")
        for qual in summary.qualifications:
            print(f"- {qual}")
            
        print("\nResponsibilities:")
        for resp in summary.responsibilities:
            print(f"- {resp}")

def main():
    """Main entry point for the script"""
    reader = JDReader("jd_raw/job_description.csv")
    summaries = reader.summarize_jobs(model='llama3.1')
    
    if summaries:
        # Display summaries
        display_summaries(summaries)
        
        # Save to SQLite
        save_to_sqlite(summaries)

if __name__ == "__main__":
    main()