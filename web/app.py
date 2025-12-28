# web/app.py

"""
Web Interface for Multi-Agent Planner

Simple Flask web application that provides a user-friendly interface
for generating code using the multi-agent system.
"""

import os
import sys
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from threading import Thread
import zipfile
import io

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import run_pipeline
from core.task_schema import Task

app = Flask(__name__)

# Store running jobs (in production, use Redis or a database)
jobs = {}


@app.route('/')
def index():
    """Main page with the input form."""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Start code generation from a project description.
    Returns immediately with a job ID.
    """
    data = request.json
    description = data.get('description', '').strip()
    
    if not description:
        return jsonify({'error': 'Description is required'}), 400
    
    # Create a job ID
    job_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Store job status
    jobs[job_id] = {
        'status': 'running',
        'description': description,
        'started_at': datetime.now().isoformat(),
        'result': None,
        'error': None
    }
    
    # Run generation in background thread
    def run_generation():
        try:
            task = Task(id=0, description=description)
            result = run_pipeline(task, save_path=f"output/session_{job_id}.json")
            
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = {
                'final_code': result,
                'test_file': read_file_safe('output/test_program.py'),
                'readme': read_file_safe('output/README.md'),
            }
        except Exception as e:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
    
    thread = Thread(target=run_generation)
    thread.start()
    
    return jsonify({'job_id': job_id, 'status': 'running'})


@app.route('/api/status/<job_id>')
def job_status(job_id):
    """Check the status of a running job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'job_id': job_id,
        'status': job['status'],
        'description': job['description'],
        'started_at': job['started_at']
    }
    
    if job['status'] == 'completed':
        response['result'] = job['result']
    elif job['status'] == 'failed':
        response['error'] = job['error']
    
    return jsonify(response)


@app.route('/api/download/<job_id>')
def download_project(job_id):
    """Download the generated project as a ZIP file."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed'}), 400
    
    # Create ZIP file in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add main program
        if job['result'].get('final_code'):
            zf.writestr('main.py', job['result']['final_code'])
        
        # Add test file
        if job['result'].get('test_file'):
            zf.writestr('test_main.py', job['result']['test_file'])
        
        # Add README
        if job['result'].get('readme'):
            zf.writestr('README.md', job['result']['readme'])
        
        # Add multi-file project if exists
        project_dir = 'output/project'
        if os.path.exists(project_dir):
            for filename in os.listdir(project_dir):
                filepath = os.path.join(project_dir, filename)
                if os.path.isfile(filepath):
                    with open(filepath, 'r') as f:
                        zf.writestr(f'project/{filename}', f.read())
    
    memory_file.seek(0)
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'generated_project_{job_id}.zip'
    )


@app.route('/api/recent')
def recent_jobs():
    """Get list of recent jobs."""
    recent = []
    for job_id, job in sorted(jobs.items(), key=lambda x: x[1]['started_at'], reverse=True)[:10]:
        recent.append({
            'job_id': job_id,
            'status': job['status'],
            'description': job['description'][:100] + ('...' if len(job['description']) > 100 else ''),
            'started_at': job['started_at']
        })
    return jsonify(recent)


def read_file_safe(filepath):
    """Read a file safely, returning None if it doesn't exist."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except:
        return None


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    print("🚀 Starting Multi-Agent Planner Web Interface...")
    print("📍 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
