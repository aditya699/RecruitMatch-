"""
Candidate Email Sender

Reads match data from database, generates personalized emails, and sends them.
"""

import os
import sqlite3
import smtplib
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ollama import chat
from datetime import datetime, timedelta

def get_top_candidate_from_db(db_file="match_results.db"):
    """Get a top matching candidate from the database"""
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get a top candidate (rank 1) for any job
        cursor.execute("""
            SELECT job_title, candidate_name, match_score 
            FROM match_results 
            WHERE rank = 1 
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "job_title": result[0],
                "candidate_name": result[1],
                "match_score": result[2]
            }
        else:
            # Fallback if no results in DB
            return {
                "job_title": "Python Developer",
                "candidate_name": "John Smith",
                "match_score": 0.89
            }
            
    except Exception as e:
        print(f"Database error: {str(e)}")
        # Fallback data
        return {
            "job_title": "Python Developer",
            "candidate_name": "John Smith",
            "match_score": 0.89
        }

def generate_dates(num_dates=3):
    """Generate upcoming interview dates (weekdays only)"""
    dates = []
    current_date = datetime.now() + timedelta(days=3)
    
    while len(dates) < num_dates:
        if current_date.weekday() < 5:  # Skip weekends
            dates.append(current_date.strftime("%A, %B %d, %Y"))
        current_date += timedelta(days=1)
    
    return dates

def draft_email(candidate_name, job_title, match_score):
    """Draft personalized email content using Ollama"""
    dates = generate_dates()
    times = ["10:00 AM", "2:00 PM", "4:30 PM"]
    
    prompt = f"""
    Draft a professional email to invite a candidate for a job interview.
    
    Details:
    - Candidate name: {candidate_name}
    - Job position: {job_title}
    - Match score: {match_score:.2f}
    - Interview dates: {', '.join(dates)}
    - Interview times: {', '.join(times)}
    - The interview will be via video conference (45 minutes)
    
    The email should be professional and concise.
    """
    
    try:
        response = chat(
            messages=[{'role': 'user', 'content': prompt}],
            model='llama3.1',
        )
        return response.message.content
    except Exception as e:
        print(f"Error generating email: {str(e)}")
        return f"""
        Subject: Interview Invitation for {job_title} Position
        
        Dear {candidate_name},
        
        We're inviting you to interview for the {job_title} position. Based on your qualifications, you're an excellent match.
        
        Dates: {', '.join(dates)}
        Times: {', '.join(times)}
        
        The interview will be via video conference (45 minutes).
        
        Please reply with your preferred date and time.
        
        Best regards,
        Recruitment Team
        """

def send_email(recipient_email, body, sender_email="ab0358031@gmail.com"):
    """Send email using Gmail SMTP"""
    sender_email = sender_email or "your_email@gmail.com"  # Replace in production
    sender_password = os.environ.get('app_password_nsp')
    
    if not sender_password:
        print("Error: APP_PASSWORD_NSP environment variable not set")
        return False
    
    # Extract subject line
    lines = body.split('\n')
    subject = next((line[8:].strip() for line in lines if line.lower().startswith('subject:')), 
                 "Interview Invitation")
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def main():
    """Main function to send a single interview email"""
    # Get candidate data from database
    candidate = get_top_candidate_from_db()
    recipient_email = "ab0358031@gmail.com"
    
    print(f"Selected candidate: {candidate['candidate_name']} for {candidate['job_title']}")
    
    # Generate email content
    email_content = draft_email(
        candidate['candidate_name'], 
        candidate['job_title'], 
        candidate['match_score']
    )
    
    # Send email
    print(f"Sending interview invitation to {candidate['candidate_name']}...")
    send_email(recipient_email, email_content)

if __name__ == "__main__":
    main()