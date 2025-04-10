# RecruitMatch: AI-Powered Recruitment System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"/>
  <img src="https://img.shields.io/badge/Flask-2.0+-red.svg" alt="Flask 2.0+"/>
  <img src="https://img.shields.io/badge/LLaMA-3.1-purple.svg" alt="LLaMA 3.1"/>
</div>

<p align="center">
  <img src="docs/recruitMatch_logo.png" alt="RecruitMatch Logo" width="300"/>
</p>

## 🚀 Overview

RecruitMatch is an end-to-end AI-powered recruitment solution that automates the entire process from job description analysis to interview scheduling. The system leverages advanced natural language processing to match candidates with job openings based on actual skills and qualifications rather than simple keyword matching.

### 🔍 Key Features

- **Job Description Analysis**: Automatically extracts key requirements, skills, and qualifications
- **CV Data Extraction**: Processes candidate resumes to identify experience, education, and skills
- **AI-Powered Matching**: Uses semantic embeddings to calculate match scores between candidates and jobs
- **Automated Interview Scheduling**: Generates personalized email invitations for top candidates
- **User-Friendly Interface**: Intuitive web dashboard for uploading files and viewing results
- **Privacy-Focused**: All processing happens locally, keeping sensitive candidate data secure

## 📋 System Architecture

RecruitMatch uses a multi-agent system architecture where specialized components handle different aspects of the recruitment workflow:

1. **JD Analyzer Agent**: Processes job description files
2. **CV Processor Agent**: Extracts structured data from candidate resumes
3. **Matching Engine**: Compares candidates to job requirements
4. **Interview Scheduler**: Handles communication with candidates
5. **Web Interface**: Provides an intuitive dashboard for recruiters

## 💻 Technology Stack

- **Backend**: Python, Flask
- **NLP**: LLaMA 3.1 via Ollama
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Document Parsing**: PyPDF
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Email**: SMTP

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Ollama with LLaMA 3.1 model installed
- Git

### Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/yourusername/recruitMatch.git
cd recruitMatch
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Make sure Ollama is running with LLaMA 3.1
```bash
ollama run llama3.1
```

5. Start the application
```bash
python simplified_server.py
```

6. Access the web interface at http://localhost:8888

## 📊 Usage

### Upload Job Descriptions

1. Navigate to the home page
2. Click on "Upload JD" and select your CSV file
3. The file should contain 'Job Title' and 'Job Description' columns

### Upload CVs

1. After uploading job descriptions, click on "Upload CVs"
2. Select multiple PDF files containing candidate resumes
3. The system will process and match candidates to jobs

### Search Candidates

1. Click on "Search Candidates" in the navigation
2. Enter a custom job title and description
3. The system will return matching candidates from the database

### View Results

1. Review match scores and candidate rankings
2. Click on "Schedule" to send interview invitations
3. Enter candidate email and customize the invitation

## 📁 Project Structure

```
recruitMatch/
│
├── cv.py                  # CV data extraction module
├── jd.py                  # Job description analysis module
├── match.py               # Candidate-job matching algorithm
├── email_send.py          # Email generation and sending
├── simplified_server.py   # Flask web application
│
├── templates/
│   ├── simple_index.html  # Main dashboard
│   ├── search_form.html   # Search interface
│   ├── search_results.html# Search results page
│   └── view_matches.html  # Match results display
│
├── static/
│   ├── css/               # Stylesheets
│   ├── js/                # JavaScript files
│   └── img/               # Images and icons
│
├── uploads/               # Temporary file storage
├── cv_raw/                # CV storage directory
├── jd_raw/                # JD storage directory
│
└── README.md              # This file
```

## 🌟 Future Enhancements

- Integration with job boards for automatic JD import
- Calendar API integration for direct interview scheduling
- Advanced analytics dashboard for recruitment metrics


## 👨‍💻 Author

**Aditya Bhatt**

- LinkedIn: [https://www.linkedin.com/in/adityaabhatt/](https://www.linkedin.com/in/adityaabhatt/)
- YouTube: [https://www.youtube.com/@adityabhatt4173](https://www.youtube.com/@adityabhatt4173)
- Instagram: [https://www.instagram.com/your_data_scientist/](https://www.instagram.com/your_data_scientist/)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

