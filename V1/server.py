"""
Recruitment System Interface

A Flask-based web interface for CV processing, job matching, and interview scheduling.
Integrates with existing CV processor, JD analyzer, and matching modules.
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import pandas as pd
import sqlite3
import json
import datetime
from werkzeug.utils import secure_filename

# Import existing modules
import cv as cv_module
import jd as jd_module
import match as match_module
from email_send import draft_email, send_email, generate_dates

app = Flask(__name__)
app.secret_key = "recruitment_system_secret_key"

# Configuration
UPLOAD_FOLDER = 'uploads'
BASE_CV_FOLDER = 'cv_raw'  # Base folder name for CV storage
BASE_JD_FOLDER = 'jd_raw'  # Base folder name for JD storage
ALLOWED_EXTENSIONS_CV = {'pdf'}
ALLOWED_EXTENSIONS_JD = {'csv'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_CV_FOLDER, exist_ok=True)
os.makedirs(BASE_JD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def create_timestamped_folder(base_folder):
    """Create a timestamped folder for this upload session"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{base_folder}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def allowed_file(filename, allowed_extensions):
    """Check if filename has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/upload_cv', methods=['GET', 'POST'])
def upload_cv():
    """Handle CV file uploads"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'cv_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        files = request.files.getlist('cv_file')
        
        if not files or files[0].filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Create a timestamped CV folder for this batch
        cv_folder = create_timestamped_folder(BASE_CV_FOLDER)
            
        # Process all uploaded files
        upload_count = 0
        for file in files:
            if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_CV):
                filename = secure_filename(file.filename)
                file_path = os.path.join(cv_folder, filename)
                file.save(file_path)
                upload_count += 1
        
        if upload_count > 0:
            flash(f'Successfully uploaded {upload_count} CV files to {cv_folder}')
            
            # Process the new CVs
            processor = cv_module.CVProcessor(cv_folder)
            cv_data = processor.process_all_cvs()
            
            if cv_data:
                cv_module.save_to_csv(cv_data)
                cv_module.save_to_sqlite(cv_data)
                flash(f'Processed {len(cv_data)} CVs')
            
        return redirect(url_for('index'))
        
    return render_template('upload_cv.html')

@app.route('/upload_jd', methods=['GET', 'POST'])
def upload_jd():
    """Handle job description CSV file upload"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'jd_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['jd_file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_JD):
            # Create a timestamped JD folder for this batch
            jd_folder = create_timestamped_folder(BASE_JD_FOLDER)
            
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(jd_folder, filename)
            file.save(file_path)
            
            flash(f'Job description file uploaded successfully to {jd_folder}')
            
            # Process the job descriptions
            reader = jd_module.JDReader(file_path)
            summaries = reader.summarize_jobs()
            
            if summaries:
                # Create session-specific database file
                db_file = f"job_summaries_{os.path.basename(jd_folder)}.db"
                jd_module.save_to_sqlite(summaries, db_file=db_file)
                flash(f'Processed {len(summaries)} job descriptions')
                
                # Also save to the main database for historical purposes
                jd_module.save_to_sqlite(summaries)
                
                # Store this upload session info
                session_info = {
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'cv_folder': request.args.get('cv_folder', 'None'),
                    'jd_folder': jd_folder,
                    'jd_count': len(summaries),
                    'db_file': db_file
                }
                
                # Save session info to file
                sessions_file = "upload_sessions.json"
                try:
                    if os.path.exists(sessions_file):
                        with open(sessions_file, 'r') as f:
                            sessions = json.load(f)
                    else:
                        sessions = []
                    
                    sessions.append(session_info)
                    
                    with open(sessions_file, 'w') as f:
                        json.dump(sessions, f, indent=2)
                except Exception as e:
                    flash(f'Error saving session info: {str(e)}')
                
                # Create or update match results
                try:
                    # Create a custom job data DataFrame just for this session
                    job_data = pd.DataFrame({
                        'Job Title': [s['job_title'] for s in summaries],
                        'Skills': [', '.join(s['summary'].skills) for s in summaries],
                        'Experience': [s['summary'].experience for s in summaries],
                        'Qualifications': [', '.join(s['summary'].qualifications) for s in summaries],
                        'Responsibilities': [', '.join(s['summary'].responsibilities) for s in summaries]
                    })
                    
                    # Load latest CV data
                    cv_data = pd.read_csv("cv_data.csv")
                    
                    if cv_data is not None and not job_data.empty:
                        # Process data for matching
                        cv_texts = cv_data.apply(lambda row: match_module.create_composite_text(row, is_cv=True), axis=1)
                        job_texts = job_data.apply(lambda row: match_module.create_composite_text(row, is_cv=False), axis=1)
                        
                        # Get embeddings
                        print("Getting embeddings for CVs...")
                        cv_embeddings = []
                        for i, text in enumerate(cv_texts):
                            print(f"  Processing CV {i+1}/{len(cv_texts)}")
                            embedding = match_module.get_embeddings(text)
                            if embedding:
                                cv_embeddings.append(embedding)
                            else:
                                cv_embeddings.append([0] * 3072)
                        
                        print("Getting embeddings for jobs...")
                        job_embeddings = []
                        for i, text in enumerate(job_texts):
                            print(f"  Processing job {i+1}/{len(job_texts)}")
                            embedding = match_module.get_embeddings(text)
                            if embedding:
                                job_embeddings.append(embedding)
                            else:
                                job_embeddings.append([0] * 3072)
                        
                        # Calculate matches
                        from sklearn.metrics.pairwise import cosine_similarity
                        import numpy as np
                        
                        # Calculate cosine similarity
                        similarity_matrix = cosine_similarity(np.array(cv_embeddings), np.array(job_embeddings))
                        
                        # For each job, find top 5 CV matches
                        results = []
                        for job_idx in range(len(job_data)):
                            job_title = job_data.iloc[job_idx]['Job Title']
                            
                            # Get scores for this job
                            job_scores = similarity_matrix[:, job_idx]
                            
                            # Get indices of top 5 matches
                            top_indices = np.argsort(job_scores)[-5:][::-1]
                            
                            job_matches = []
                            for rank, cv_idx in enumerate(top_indices):
                                match_score = job_scores[cv_idx]
                                # Skip very low match scores (less than 0.1)
                                if match_score < 0.1:
                                    continue
                                    
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
                        
                        # Save to session-specific match results
                        match_results_file = f"match_results_{os.path.basename(jd_folder)}.db"
                        match_module.save_match_results(results, db_file=match_results_file)
                        
                        # Also save to main match results for consistency
                        match_module.save_match_results(results)
                        
                        flash(f'Created match results for this upload session: {match_results_file}')
                except Exception as e:
                    flash(f'Error updating matches: {str(e)}')
            
            return redirect(url_for('index'))
            
    return render_template('upload_jd.html')

@app.route('/matches')
def view_matches():
    """View job-candidate matches"""
    try:
        # Connect to DB and get unique job titles
        conn = sqlite3.connect('match_results.db')
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT job_title FROM match_results")
        job_titles = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return render_template('matches.html', job_titles=job_titles)
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(url_for('index'))

@app.route('/upload_sessions')
def view_sessions():
    """View upload sessions and their corresponding match results"""
    try:
        # Load session data from JSON file
        sessions_file = "upload_sessions.json"
        if os.path.exists(sessions_file):
            with open(sessions_file, 'r') as f:
                sessions = json.load(f)
        else:
            sessions = []
        
        return render_template('sessions.html', sessions=sessions)
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(url_for('index'))

@app.route('/api/matches/<job_title>')
def get_matches(job_title):
    """API endpoint to get matches for a specific job title"""
    try:
        conn = sqlite3.connect('match_results.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get top 5 matches for the job title
        cursor.execute("""
            SELECT job_title, rank, candidate_name, cv_filename, match_score 
            FROM match_results 
            WHERE job_title = ? 
            ORDER BY match_score DESC
            LIMIT 5
        """, (job_title,))
        
        matches = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify(matches)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/session_matches/<session_db>')
def get_session_matches(session_db):
    """API endpoint to get matches from a specific session database"""
    try:
        if not os.path.exists(session_db):
            return jsonify({"error": "Session database not found"}), 404
            
        conn = sqlite3.connect(session_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all unique job titles in this session
        cursor.execute("SELECT DISTINCT job_title FROM match_results")
        job_titles = [row[0] for row in cursor.fetchall()]
        
        all_matches = {}
        
        # For each job title, get top matches
        for job_title in job_titles:
            cursor.execute("""
                SELECT job_title, rank, candidate_name, cv_filename, match_score 
                FROM match_results 
                WHERE job_title = ? 
                ORDER BY match_score DESC
            """, (job_title,))
            
            matches = [dict(row) for row in cursor.fetchall()]
            all_matches[job_title] = matches
        
        conn.close()
        
        return jsonify(all_matches)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/session_matches/<session_id>')
def view_session_matches(session_id):
    """View matches for a specific upload session"""
    try:
        # Load session data from JSON file
        sessions_file = "upload_sessions.json"
        if os.path.exists(sessions_file):
            with open(sessions_file, 'r') as f:
                sessions = json.load(f)
        else:
            sessions = []
        
        # Find the specific session
        session = next((s for s in sessions if s.get('jd_folder', '').endswith(session_id)), None)
        
        if not session:
            flash('Session not found')
            return redirect(url_for('view_sessions'))
        
        return render_template('session_matches.html', session=session)
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(url_for('view_sessions'))

@app.route('/search_candidates', methods=['GET', 'POST'])
def search_candidates():
    """Search for candidates based on job description text input"""
    if request.method == 'POST':
        job_title = request.form.get('job_title')
        job_description = request.form.get('job_description')
        
        if not job_title or not job_description:
            flash('Please provide both job title and description')
            return redirect(request.url)
        
        # Save job description to temporary CSV
        temp_jd_file = os.path.join(BASE_JD_FOLDER, 'temp_job.csv')
        df = pd.DataFrame({
            'Job Title': [job_title],
            'Job Description': [job_description]
        })
        df.to_csv(temp_jd_file, index=False)
        
        # Process the job description
        reader = jd_module.JDReader(temp_jd_file)
        summaries = reader.summarize_jobs()
        
        if summaries:
            # Save to temporary DB to avoid affecting main data
            jd_module.save_to_sqlite(summaries, db_file="temp_job_summary.db")
            
            # Load CV data
            cv_data = pd.read_csv("cv_data.csv")
            
            # Create job data DataFrame from the processed summary
            job_data = pd.DataFrame({
                'Job Title': [job_title],
                'Skills': [', '.join(summaries[0]['summary'].skills)],
                'Experience': [summaries[0]['summary'].experience],
                'Qualifications': [', '.join(summaries[0]['summary'].qualifications)],
                'Responsibilities': [', '.join(summaries[0]['summary'].responsibilities)]
            })
            
            # Process for matching
            cv_texts = cv_data.apply(lambda row: match_module.create_composite_text(row, is_cv=True), axis=1)
            job_texts = job_data.apply(lambda row: match_module.create_composite_text(row, is_cv=False), axis=1)
            
            # Get embeddings
            cv_embeddings = []
            for text in cv_texts:
                embedding = match_module.get_embeddings(text)
                if embedding:
                    cv_embeddings.append(embedding)
                else:
                    cv_embeddings.append([0] * 3072)
            
            job_embedding = match_module.get_embeddings(job_texts.iloc[0])
            if not job_embedding:
                job_embedding = [0] * 3072
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            similarity_scores = cosine_similarity([job_embedding], cv_embeddings)[0]
            
            # Get top 5 matches
            top_indices = np.argsort(similarity_scores)[-5:][::-1]
            
            matches = []
            for rank, cv_idx in enumerate(top_indices):
                match_score = similarity_scores[cv_idx]
                matches.append({
                    'rank': rank + 1,
                    'job_title': job_title,
                    'candidate_name': cv_data.iloc[cv_idx]['Name'],
                    'cv_filename': cv_data.iloc[cv_idx]['Filename'],
                    'match_score': float(match_score)
                })
            
            return render_template('search_results.html', matches=matches, job_title=job_title)
        
        flash('Failed to process job description')
        return redirect(request.url)
    
    return render_template('search_candidates.html')

@app.route('/schedule_interview/<job_title>/<candidate_name>', methods=['GET', 'POST'])
def schedule_interview(job_title, candidate_name):
    """Schedule an interview with a matched candidate"""
    if request.method == 'POST':
        # Get form data
        email = request.form.get('email')
        # In production, get real match score from database
        match_score = float(request.form.get('match_score', 0.85))
        
        # Generate email using existing module
        email_content = draft_email(candidate_name, job_title, match_score)
        
        # Send email
        if send_email(email, email_content):
            flash('Interview invitation sent successfully!')
        else:
            flash('Failed to send email invitation')
        
        return redirect(url_for('view_matches'))
    
    # Get available dates
    dates = generate_dates()
    
    return render_template(
        'schedule_interview.html', 
        job_title=job_title,
        candidate_name=candidate_name,
        dates=dates
    )

@app.route('/api/send_invitation', methods=['POST'])
def send_invitation():
    """API endpoint to send an interview invitation"""
    data = request.json
    candidate_name = data.get('candidate_name')
    job_title = data.get('job_title')
    email = data.get('email')
    match_score = float(data.get('match_score', 0.85))
    
    # Generate and send email
    email_content = draft_email(candidate_name, job_title, match_score)
    success = send_email(email, email_content)
    
    return jsonify({"success": success})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)