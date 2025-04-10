"""
Simplified Recruitment System Interface

A streamlined Flask application focused on two core features:
1. Upload JD and CV in the same session and show matches
2. Search CV database with custom job description
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import pandas as pd
import sqlite3
import uuid
from werkzeug.utils import secure_filename

# Import existing modules
import cv as cv_module
import jd as jd_module
import match as match_module
from email_send import draft_email, send_email

app = Flask(__name__)
app.secret_key = "recruitment_system_secret_key"

# Configuration
UPLOAD_FOLDER = 'uploads'
CV_FOLDER = 'cv_raw'
JD_FOLDER = 'jd_raw'
ALLOWED_EXTENSIONS_CV = {'pdf'}
ALLOWED_EXTENSIONS_JD = {'csv'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CV_FOLDER, exist_ok=True)
os.makedirs(JD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'

def allowed_file(filename, allowed_extensions):
    """Check if filename has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """Render the main page with both upload forms"""
    return render_template('simple_index.html')

@app.route('/upload_jd', methods=['POST'])
def upload_jd():
    """Handle job description CSV file upload"""
    if 'jd_file' not in request.files:
        flash('No JD file selected')
        return redirect(url_for('index'))
        
    file = request.files['jd_file']
    
    if file.filename == '':
        flash('No JD file selected')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_JD):
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(JD_FOLDER, filename)
        file.save(file_path)
        
        # Store the JD path in session
        session['jd_path'] = file_path
        
        flash('Job description uploaded successfully. Now upload CVs to see matches.')
        return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a CSV file.')
    return redirect(url_for('index'))

@app.route('/upload_cv', methods=['POST'])
def upload_cv():
    """Handle CV file uploads and process matches"""
    if 'cv_files' not in request.files:
        flash('No CV files selected')
        return redirect(url_for('index'))
        
    files = request.files.getlist('cv_files')
    
    if not files or files[0].filename == '':
        flash('No CV files selected')
        return redirect(url_for('index'))
    
    # Check if JD was uploaded first
    if 'jd_path' not in session:
        flash('Please upload a job description first')
        return redirect(url_for('index'))
    
    # Process uploaded CV files
    cv_paths = []
    for file in files:
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_CV):
            filename = secure_filename(file.filename)
            file_path = os.path.join(CV_FOLDER, filename)
            file.save(file_path)
            cv_paths.append(file_path)
    
    if not cv_paths:
        flash('No valid CV files uploaded')
        return redirect(url_for('index'))
    
    # Process JD
    jd_path = session['jd_path']
    reader = jd_module.JDReader(jd_path)
    jd_summaries = reader.summarize_jobs()
    
    if not jd_summaries:
        flash('Failed to process job descriptions')
        return redirect(url_for('index'))
    
    # Process CVs
    cv_data = []
    for cv_path in cv_paths:
        processor = cv_module.CVProcessor(os.path.dirname(cv_path))
        result = processor.extract_cv_data(cv_path)
        if result:
            cv_data.append({
                'filename': os.path.basename(cv_path),
                'data': result
            })
    
    if not cv_data:
        flash('Failed to process CVs')
        return redirect(url_for('index'))
    
    # Create dataframes for matching
    # Convert JD summaries to DataFrame
    job_data = pd.DataFrame({
        'Job Title': [s['job_title'] for s in jd_summaries],
        'Skills': [', '.join(s['summary'].skills) for s in jd_summaries],
        'Experience': [s['summary'].experience for s in jd_summaries],
        'Qualifications': [', '.join(s['summary'].qualifications) for s in jd_summaries],
        'Responsibilities': [', '.join(s['summary'].responsibilities) for s in jd_summaries]
    })
    
    # Convert CV data to DataFrame format compatible with matching function
    cv_df_data = {
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
    
    for item in cv_data:
        cv = item['data']
        cv_df_data['Filename'].append(item['filename'])
        cv_df_data['Name'].append(cv.name)
        cv_df_data['Email'].append(cv.email)
        cv_df_data['Phone'].append(cv.phone)
        cv_df_data['Skills'].append(', '.join(cv.skills))
        
        # Format education
        education_items = []
        for edu in cv.education:
            education_items.append(f"{edu.degree} from {edu.institution} ({edu.year})")
        cv_df_data['Education'].append('; '.join(education_items))
        
        # Format experience
        experience_items = []
        for exp in cv.experience:
            experience_items.append(f"{exp.position} at {exp.company} ({exp.duration})")
        cv_df_data['Experience'].append('; '.join(experience_items))
        
        cv_df_data['Certifications'].append(', '.join(cv.certifications))
        cv_df_data['Languages'].append(', '.join(cv.languages) if cv.languages else '')
    
    cv_df = pd.DataFrame(cv_df_data)
    
    # Calculate matches
    matches = []
    try:
        # Create texts for embedding
        cv_texts = cv_df.apply(lambda row: match_module.create_composite_text(row, is_cv=True), axis=1)
        job_texts = job_data.apply(lambda row: match_module.create_composite_text(row, is_cv=False), axis=1)
        
        # Get embeddings
        cv_embeddings = []
        for text in cv_texts:
            embedding = match_module.get_embeddings(text)
            if embedding:
                cv_embeddings.append(embedding)
            else:
                cv_embeddings.append([0] * 3072)  # Default embedding size
        
        job_embeddings = []
        for text in job_texts:
            embedding = match_module.get_embeddings(text)
            if embedding:
                job_embeddings.append(embedding)
            else:
                job_embeddings.append([0] * 3072)  # Default embedding size
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        similarity_matrix = cosine_similarity(np.array(cv_embeddings), np.array(job_embeddings))
        
        # Process results
        for job_idx, job_title in enumerate(job_data['Job Title']):
            job_scores = similarity_matrix[:, job_idx]
            
            # Sort CV indices by score (descending)
            sorted_indices = np.argsort(job_scores)[::-1]
            
            # Take top matches
            for rank, cv_idx in enumerate(sorted_indices[:5]):
                match_score = job_scores[cv_idx]
                
                # Skip very low scores
                if match_score < 0.1:
                    continue
                    
                matches.append({
                    'job_title': job_title,
                    'rank': rank + 1,
                    'candidate_name': cv_df.iloc[cv_idx]['Name'],
                    'cv_filename': cv_df.iloc[cv_idx]['Filename'],
                    'match_score': float(match_score)
                })
    except Exception as e:
        flash(f'Error calculating matches: {str(e)}')
        return redirect(url_for('index'))
    
    # Store matches in session
    session['matches'] = matches
    
    return redirect(url_for('view_matches'))

@app.route('/view_matches')
def view_matches():
    """View the matches from the current session"""
    if 'matches' not in session:
        flash('No matches found. Please upload files first.')
        return redirect(url_for('index'))
    
    matches = session['matches']
    
    # Group matches by job title
    grouped_matches = {}
    for match in matches:
        job_title = match['job_title']
        if job_title not in grouped_matches:
            grouped_matches[job_title] = []
        grouped_matches[job_title].append(match)
    
    # Sort each group by match score
    for job_title in grouped_matches:
        grouped_matches[job_title].sort(key=lambda x: x['match_score'], reverse=True)
    
    return render_template('view_matches.html', grouped_matches=grouped_matches)

@app.route('/search', methods=['GET', 'POST'])
def search_candidates():
    """Search for candidates based on job description text input"""
    if request.method == 'POST':
        job_title = request.form.get('job_title')
        job_description = request.form.get('job_description')
        
        if not job_title or not job_description:
            flash('Please provide both job title and description')
            return redirect(url_for('search_candidates'))
        
        # Save job description to temporary CSV
        temp_file = os.path.join(JD_FOLDER, 'temp_search.csv')
        pd.DataFrame({
            'Job Title': [job_title],
            'Job Description': [job_description]
        }).to_csv(temp_file, index=False)
        
        # Process job description
        reader = jd_module.JDReader(temp_file)
        summaries = reader.summarize_jobs()
        
        if not summaries:
            flash('Failed to process job description')
            return redirect(url_for('search_candidates'))
        
        # Check if we have CV data
        try:
            cv_data = pd.read_csv("cv_data.csv")
            if cv_data.empty:
                flash('No CV data available for matching')
                return redirect(url_for('search_candidates'))
        except:
            flash('No CV database found. Please upload CVs first.')
            return redirect(url_for('search_candidates'))
        
        # Create job data DataFrame
        job_data = pd.DataFrame({
            'Job Title': [job_title],
            'Skills': [', '.join(summaries[0]['summary'].skills)],
            'Experience': [summaries[0]['summary'].experience],
            'Qualifications': [', '.join(summaries[0]['summary'].qualifications)],
            'Responsibilities': [', '.join(summaries[0]['summary'].responsibilities)]
        })
        
        # Calculate matches
        matches = []
        try:
            # Create texts for embedding
            cv_texts = cv_data.apply(lambda row: match_module.create_composite_text(row, is_cv=True), axis=1)
            job_text = match_module.create_composite_text(job_data.iloc[0], is_cv=False)
            
            # Get embeddings
            cv_embeddings = []
            for text in cv_texts:
                embedding = match_module.get_embeddings(text)
                if embedding:
                    cv_embeddings.append(embedding)
                else:
                    cv_embeddings.append([0] * 3072)
            
            job_embedding = match_module.get_embeddings(job_text)
            if not job_embedding:
                job_embedding = [0] * 3072
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            similarity_scores = cosine_similarity([job_embedding], np.array(cv_embeddings))[0]
            
            # Sort CV indices by score (descending)
            sorted_indices = np.argsort(similarity_scores)[::-1]
            
            # Take top matches
            for rank, cv_idx in enumerate(sorted_indices[:10]):
                match_score = similarity_scores[cv_idx]
                
                # Skip very low scores
                if match_score < 0.1:
                    continue
                    
                matches.append({
                    'job_title': job_title,
                    'rank': rank + 1,
                    'candidate_name': cv_data.iloc[cv_idx]['Name'],
                    'cv_filename': cv_data.iloc[cv_idx]['Filename'],
                    'match_score': float(match_score)
                })
        except Exception as e:
            flash(f'Error calculating matches: {str(e)}')
            return redirect(url_for('search_candidates'))
        
        return render_template('search_results.html', 
                              matches=matches, 
                              job_title=job_title,
                              job_description=job_description)
    
    return render_template('search_form.html')

@app.route('/send_invitation', methods=['POST'])
def send_invitation():
    """Send interview invitation email"""
    candidate_name = request.form.get('candidate_name')
    job_title = request.form.get('job_title')
    email = request.form.get('email')
    match_score = float(request.form.get('match_score', 0.85))
    
    if not candidate_name or not job_title or not email:
        flash('Missing required information')
        return redirect(request.referrer or url_for('index'))
    
    # Generate email content
    email_content = draft_email(candidate_name, job_title, match_score)
    
    # Send email
    success = send_email(email, email_content)
    
    if success:
        flash(f'Interview invitation sent to {candidate_name} for {job_title} position')
    else:
        flash('Failed to send interview invitation')
    
    return redirect(request.referrer or url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)