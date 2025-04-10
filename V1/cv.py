"""
CV Data Extractor

Extracts key data from CVs/resumes in PDF format:
- Education
- Work experience
- Skills
- Certifications
- Other relevant information

Author: Aditya Bhatt
"""

import os
import pandas as pd
import sqlite3
from langchain_community.document_loaders import PyPDFLoader
from ollama import chat
from pydantic import BaseModel
from typing import List, Optional, Dict

class Education(BaseModel):
    """Education details from CV"""
    degree: str
    institution: str
    year: str

class Experience(BaseModel):
    """Work experience details from CV"""
    company: str
    position: str
    duration: str
    description: str

class CVData(BaseModel):
    """Structured CV data model"""
    name: str
    email: Optional[str]
    phone: Optional[str]
    skills: List[str]
    education: List[Education]
    experience: List[Experience]
    certifications: List[str]
    languages: Optional[List[str]]
    
class CVProcessor:
    """Process CVs from PDF files and extract structured information"""
    
    def __init__(self, folder_path):
        """
        Initialize with path to folder containing CV PDF files
        
        Args:
            folder_path (str): Path to directory with CV PDFs
        """
        self.folder_path = folder_path
        
    def get_cv_files(self):
        """
        Get list of PDF files in the specified folder
        
        Returns:
            list: List of PDF file paths
        """
        pdf_files = []
        
        try:
            for file in os.listdir(self.folder_path):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(self.folder_path, file))
            
            print(f"Found {len(pdf_files)} PDF files")
            return pdf_files
        
        except Exception as e:
            print(f"Error accessing folder: {str(e)}")
            return []
    
    def extract_cv_data(self, file_path, model='llama3.1'):
        """
        Extract structured data from a CV PDF file
        
        Args:
            file_path (str): Path to CV PDF file
            model (str): Ollama model name
            
        Returns:
            CVData: Structured CV data
            None: If extraction fails
        """
        try:
            # Load PDF document
            print(f"Processing: {os.path.basename(file_path)}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Combine all pages into a single text
            full_text = ""
            for doc in docs:
                full_text += doc.page_content + "\n\n"
            
            # Create prompt for the LLM
            prompt = f"""
            Extract key information from the following CV/resume:
            
            {full_text[:4000]}  # Limit text length to avoid token issues
            
            Extract name, email, phone, skills, education details (degree, institution, year), 
            work experience (company, position, duration, description), certifications, and languages.
            """
            
            # Get structured response from Ollama
            response = chat(
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ],
                model=model,
                format=CVData.model_json_schema(),
            )
            
            # Parse the response into our Pydantic model
            cv_data = CVData.model_validate_json(response.message.content)
            return cv_data
            
        except Exception as e:
            print(f"Error extracting data from {file_path}: {str(e)}")
            return None
    
    def process_all_cvs(self, model='llama3.1'):
        """
        Process all CV PDFs in the folder
        
        Args:
            model (str): Ollama model name
            
        Returns:
            list: List of CV data objects
            None: If no CVs found or processed
        """
        cv_files = self.get_cv_files()
        if not cv_files:
            return None
        
        results = []
        total_files = len(cv_files)
        
        for i, file_path in enumerate(cv_files):
            print(f"Processing CV {i+1}/{total_files}: {os.path.basename(file_path)}")
            cv_data = self.extract_cv_data(file_path, model)
            
            if cv_data:
                # Add filename to results for reference
                results.append({
                    'filename': os.path.basename(file_path),
                    'data': cv_data
                })
        
        return results

def save_to_sqlite(cv_data_list, db_file="cv_data.db"):
    """
    Save extracted CV data to SQLite database
    
    Args:
        cv_data_list (list): List of CV data dictionaries
        db_file (str): Path to SQLite database file
        
    Returns:
        None
    """
    if not cv_data_list:
        print("No CV data to save")
        return
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create table for CV data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cv_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        name TEXT,
        email TEXT,
        phone TEXT,
        skills TEXT,
        education TEXT,
        experience TEXT,
        certifications TEXT,
        languages TEXT
    )
    ''')
    
    # Insert data
    for item in cv_data_list:
        filename = item['filename']
        data = item['data']
        
        # Convert lists and objects to strings for storage
        skills_str = ', '.join(data.skills)
        certifications_str = ', '.join(data.certifications)
        languages_str = ', '.join(data.languages) if data.languages else ''
        
        # Convert education list to string
        education_items = []
        for edu in data.education:
            education_items.append(f"{edu.degree} from {edu.institution} ({edu.year})")
        education_str = '; '.join(education_items)
        
        # Convert experience list to string
        experience_items = []
        for exp in data.experience:
            experience_items.append(f"{exp.position} at {exp.company} ({exp.duration}): {exp.description}")
        experience_str = '; '.join(experience_items)
        
        cursor.execute(
            """INSERT INTO cv_data 
            (filename, name, email, phone, skills, education, experience, certifications, languages) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                filename, 
                data.name,
                data.email,
                data.phone,
                skills_str,
                education_str,
                experience_str,
                certifications_str,
                languages_str
            )
        )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"CV data saved to SQLite database: {db_file}")

def save_to_csv(cv_data_list, output_file="cv_data.csv"):
    """
    Save extracted CV data to CSV file
    
    Args:
        cv_data_list (list): List of CV data dictionaries
        output_file (str): Path to output CSV file
        
    Returns:
        DataFrame: Pandas DataFrame with CV data
    """
    if not cv_data_list:
        print("No CV data to save")
        return None
    
    # Prepare data for CSV
    data = {
        'Filename': [],
        'Name': [],
        'Email': [],
        'Phone': [],
        'Skills': [],
        'Education': [],
        'Experience': [],
        'Certifications': [],
        'Languages': []
    }
    
    for item in cv_data_list:
        filename = item['filename']
        cv = item['data']
        
        data['Filename'].append(filename)
        data['Name'].append(cv.name)
        data['Email'].append(cv.email)
        data['Phone'].append(cv.phone)
        data['Skills'].append(', '.join(cv.skills))
        
        # Format education
        education_items = []
        for edu in cv.education:
            education_items.append(f"{edu.degree} from {edu.institution} ({edu.year})")
        data['Education'].append('; '.join(education_items))
        
        # Format experience
        experience_items = []
        for exp in cv.experience:
            experience_items.append(f"{exp.position} at {exp.company} ({exp.duration})")
        data['Experience'].append('; '.join(experience_items))
        
        data['Certifications'].append(', '.join(cv.certifications))
        data['Languages'].append(', '.join(cv.languages) if cv.languages else '')
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"CV data saved to CSV: {output_file}")
    return df

def main():
    """Main entry point for the script"""
    # Use hardcoded path
    cv_folder = "cv_raw"
    
    # Process CVs
    processor = CVProcessor(cv_folder)
    cv_data = processor.process_all_cvs(model='llama3.1')
    
    if cv_data:
        # Save to CSV
        save_to_csv(cv_data)
        
        # Save to SQLite
        save_to_sqlite(cv_data)
    else:
        print("No CV data was extracted")

if __name__ == "__main__":
    main()