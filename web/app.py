# web/app.py

"""
Web Interface for Multi-Agent Planner

Simple Flask web application that provides a user-friendly interface
for generating code using the multi-agent system.

Features:
- REST API for code generation
- Rate limiting to prevent abuse
- Background job processing
- ZIP download of generated projects
"""

import io
import os
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from functools import wraps
from threading import Lock, Thread

from flask import Flask, jsonify, render_template, request, send_file

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import run_pipeline
from core.task_schema import Task

app = Flask(__name__)


# ===========================================
# Rate Limiting
# ===========================================

class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    For production, use Redis-based limiter (Flask-Limiter).
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if a request is allowed.
        
        Returns:
            (allowed: bool, remaining: int)
        """
        now = time.time()

        with self.lock:
            # Clean old requests
            self.requests[key] = [
                t for t in self.requests[key]
                if now - t < self.window_seconds
            ]

            # Check limit
            if len(self.requests[key]) >= self.max_requests:
                return False, 0

            # Record this request
            self.requests[key].append(now)
            remaining = self.max_requests - len(self.requests[key])
            return True, remaining


# Rate limiter: 10 requests per minute per IP
limiter = RateLimiter(
    max_requests=int(os.environ.get('RATE_LIMIT_MAX', 10)),
    window_seconds=int(os.environ.get('RATE_LIMIT_WINDOW', 60))
)


def rate_limit(f):
    """Decorator to apply rate limiting to routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get client IP
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if client_ip:
            client_ip = client_ip.split(',')[0].strip()

        allowed, remaining = limiter.is_allowed(client_ip)

        if not allowed:
            response = jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests. Please wait before trying again.',
                'retry_after': limiter.window_seconds
            })
            response.status_code = 429
            response.headers['Retry-After'] = str(limiter.window_seconds)
            response.headers['X-RateLimit-Remaining'] = '0'
            return response

        # Add rate limit headers to response
        response = f(*args, **kwargs)
        if hasattr(response, 'headers'):
            response.headers['X-RateLimit-Remaining'] = str(remaining)
            response.headers['X-RateLimit-Limit'] = str(limiter.max_requests)
        return response

    return decorated_function


# ===========================================
# Job Storage
# ===========================================

# Store running jobs (in production, use Redis or a database)
jobs = {}


# ===========================================
# Routes
# ===========================================

@app.route('/')
def index():
    """Main page with the input form."""
    return render_template('index.html')


@app.route('/api/docs')
def api_docs():
    """Serve OpenAPI spec as YAML."""
    return send_file(
        os.path.join(os.path.dirname(__file__), 'openapi.yml'),
        mimetype='text/yaml'
    )


@app.route('/swagger')
def swagger_ui():
    """Serve Swagger UI for API documentation."""
    swagger_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Agent Planner API</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        SwaggerUIBundle({
            url: "/api/docs",
            dom_id: '#swagger-ui',
            presets: [SwaggerUIBundle.presets.apis],
            layout: "BaseLayout"
        });
    </script>
</body>
</html>
'''
    return swagger_html


@app.route('/api/generate', methods=['POST'])
@rate_limit
def generate():
    """
    Start code generation from a project description.
    Returns immediately with a job ID.
    
    Rate limited: 10 requests per minute per IP.
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
                    with open(filepath) as f:
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


@app.route('/api/health')
def health_check():
    """Health check endpoint for Docker/Kubernetes."""
    return jsonify({
        'status': 'healthy',
        'service': 'multi-agent-planner',
        'version': '1.0.0',
        'active_jobs': len([j for j in jobs.values() if j['status'] == 'running'])
    })


def read_file_safe(filepath):
    """Read a file safely, returning None if it doesn't exist."""
    try:
        with open(filepath) as f:
            return f.read()
    except:
        return None


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Use port 8080 to avoid conflict with macOS AirPlay (port 5000)
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print("Starting Multi-Agent Planner Web Interface...")
    print(f"Open http://localhost:{port} in your browser")
    app.run(debug=debug, host=host, port=port)
