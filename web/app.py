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

from flask import Flask, Response, jsonify, render_template, request, send_file

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The web app serves untrusted input: never let generated code fall back to
# the crash-isolated (non-sandboxed) subprocess executor. Must be set before
# the orchestrator/agents import. Operators can override explicitly with
# MAP_FORBID_CRASH_ISOLATED=0 for a trusted localhost-only deployment.
os.environ.setdefault("MAP_FORBID_CRASH_ISOLATED", "1")

from core.events import get_bus, new_job_id
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
            # Evict idle keys so the dict can't grow unbounded
            stale = [
                k
                for k, times in self.requests.items()
                if not times or now - times[-1] > self.window_seconds
            ]
            for k in stale:
                if k != key:
                    del self.requests[k]

            # Clean old requests
            self.requests[key] = [t for t in self.requests[key] if now - t < self.window_seconds]

            # Check limit
            if len(self.requests[key]) >= self.max_requests:
                return False, 0

            # Record this request
            self.requests[key].append(now)
            remaining = self.max_requests - len(self.requests[key])
            return True, remaining


# Rate limiter: 10 requests per minute per IP
limiter = RateLimiter(
    max_requests=int(os.environ.get("RATE_LIMIT_MAX", 10)),
    window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW", 60)),
)


def rate_limit(f):
    """Decorator to apply rate limiting to routes."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get client IP. X-Forwarded-For is client-controlled, so it is only
        # honored when the operator declares a trusted reverse proxy in front
        # (TRUST_PROXY=1); otherwise a caller could spoof a fresh IP per
        # request and bypass the limiter entirely.
        client_ip = request.remote_addr
        if os.environ.get("TRUST_PROXY", "").lower() in ("1", "true", "yes"):
            forwarded = request.headers.get("X-Forwarded-For", "")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()

        allowed, remaining = limiter.is_allowed(client_ip)

        if not allowed:
            response = jsonify(
                {
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please wait before trying again.",
                    "retry_after": limiter.window_seconds,
                }
            )
            response.status_code = 429
            response.headers["Retry-After"] = str(limiter.window_seconds)
            response.headers["X-RateLimit-Remaining"] = "0"
            return response

        # Add rate limit headers to response
        response = f(*args, **kwargs)
        if hasattr(response, "headers"):
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Limit"] = str(limiter.max_requests)
        return response

    return decorated_function


# ===========================================
# Job Storage
# ===========================================

# Store running jobs (in production, use Redis or a database).
# Guarded by jobs_lock; bounded so a long-lived server can't grow forever.
jobs = {}
jobs_lock = Lock()
MAX_STORED_JOBS = int(os.environ.get("MAX_STORED_JOBS", 100))


def _store_job(job_id: str, record: dict) -> None:
    """Insert a job record, evicting the oldest finished jobs beyond the cap."""
    with jobs_lock:
        jobs[job_id] = record
        if len(jobs) > MAX_STORED_JOBS:
            finished = [
                jid
                for jid, j in sorted(jobs.items(), key=lambda x: x[1]["started_at"])
                if j["status"] != "running"
            ]
            for jid in finished[: len(jobs) - MAX_STORED_JOBS]:
                del jobs[jid]


# ===========================================
# Routes
# ===========================================


@app.route("/")
def index():
    """Main page with the input form."""
    return render_template("index.html")


@app.route("/api/docs")
def api_docs():
    """Serve OpenAPI spec as YAML."""
    return send_file(os.path.join(os.path.dirname(__file__), "openapi.yml"), mimetype="text/yaml")


@app.route("/swagger")
def swagger_ui():
    """Serve Swagger UI for API documentation."""
    swagger_html = """
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
"""
    return swagger_html


@app.route("/api/generate", methods=["POST"])
@rate_limit
def generate():
    """
    Start code generation from a project description.
    Returns immediately with a job ID.

    Rate limited: 10 requests per minute per IP.
    """
    data = request.json
    description = data.get("description", "").strip()

    if not description:
        return jsonify({"error": "Description is required"}), 400

    # Create a job ID — opaque short hash so SSE clients can subscribe by it
    job_id = new_job_id()

    # Store job status
    _store_job(
        job_id,
        {
            "status": "running",
            "description": description,
            "started_at": datetime.now().isoformat(),
            "result": None,
            "error": None,
        },
    )

    # Run generation in background thread
    def run_generation():
        try:
            task = Task(id=0, description=description)
            result = run_pipeline(
                task,
                save_path=f"output/session_{job_id}.json",
                job_id=job_id,
            )

            with jobs_lock:
                # Populate result before flipping status so a concurrent
                # /api/status reader can't see completed-with-no-result.
                jobs[job_id]["result"] = {
                    "final_code": result,
                    "test_file": read_file_safe("output/test_program.py"),
                    "readme": read_file_safe("output/README.md"),
                }
                jobs[job_id]["status"] = "completed"
        except Exception as e:
            with jobs_lock:
                jobs[job_id]["error"] = str(e)
                jobs[job_id]["status"] = "failed"
        finally:
            # If the pipeline died before signalling end-of-stream, close the
            # event bus here so SSE subscribers don't block forever.
            get_bus().end(job_id)

    thread = Thread(target=run_generation)
    thread.start()

    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/api/stream/<job_id>")
def stream(job_id):
    """Server-Sent Events stream of pipeline events for ``job_id``.

    Subscribers receive the full event history (so reconnects are safe) and
    then live events until the run signals end-of-stream.
    """
    import json as _json

    def event_stream():
        bus = get_bus()
        # Heartbeat comment every ~15s prevents proxies from dropping idle conns.
        for event in bus.subscribe(job_id, replay=True):
            data = _json.dumps(event.to_dict())
            yield f"event: {event.type}\nid: {event.seq}\ndata: {data}\n\n"
        yield "event: end\ndata: {}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # nginx: don't buffer
    }
    return Response(event_stream(), headers=headers)


@app.route("/api/status/<job_id>")
def job_status(job_id):
    """Check the status of a running job."""
    with jobs_lock:
        job = dict(jobs[job_id]) if job_id in jobs else None
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    response = {
        "job_id": job_id,
        "status": job["status"],
        "description": job["description"],
        "started_at": job["started_at"],
    }

    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job["error"]

    return jsonify(response)


@app.route("/api/download/<job_id>")
def download_project(job_id):
    """Download the generated project as a ZIP file."""
    with jobs_lock:
        job = dict(jobs[job_id]) if job_id in jobs else None
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed"}), 400

    # Create ZIP file in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add main program
        if job["result"].get("final_code"):
            zf.writestr("main.py", job["result"]["final_code"])

        # Add test file
        if job["result"].get("test_file"):
            zf.writestr("test_main.py", job["result"]["test_file"])

        # Add README
        if job["result"].get("readme"):
            zf.writestr("README.md", job["result"]["readme"])

        # Add multi-file project if exists (read as bytes: a non-UTF-8 file
        # must not 500 the whole download)
        project_dir = "output/project"
        if os.path.exists(project_dir):
            for filename in os.listdir(project_dir):
                filepath = os.path.join(project_dir, filename)
                if os.path.isfile(filepath):
                    with open(filepath, "rb") as f:
                        zf.writestr(f"project/{filename}", f.read())

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"generated_project_{job_id}.zip",
    )


@app.route("/api/recent")
def recent_jobs():
    """Get list of recent jobs."""
    recent = []
    with jobs_lock:
        snapshot = {jid: dict(j) for jid, j in jobs.items()}
    for job_id, job in sorted(snapshot.items(), key=lambda x: x[1]["started_at"], reverse=True)[
        :10
    ]:
        recent.append(
            {
                "job_id": job_id,
                "status": job["status"],
                "description": job["description"][:100]
                + ("..." if len(job["description"]) > 100 else ""),
                "started_at": job["started_at"],
            }
        )
    return jsonify(recent)


@app.route("/api/health")
def health_check():
    """Health check endpoint for Docker/Kubernetes."""
    with jobs_lock:
        active = len([j for j in jobs.values() if j["status"] == "running"])
    return jsonify(
        {
            "status": "healthy",
            "service": "multi-agent-planner",
            "version": "2.0.0",
            "active_jobs": active,
        }
    )


def read_file_safe(filepath):
    """Read a file safely, returning None if it doesn't exist."""
    try:
        with open(filepath) as f:
            return f.read()
    except OSError:
        return None


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Use port 8080 to avoid conflict with macOS AirPlay (port 5000)
    port = int(os.environ.get("PORT", 8080))
    # Bind loopback by default; opt in to network exposure with HOST=0.0.0.0
    # (the Docker image does — the container boundary is its isolation).
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    if debug and host not in ("127.0.0.1", "localhost", "::1"):
        # The Werkzeug debugger is an RCE vector; never expose it off-box.
        print(f"FLASK_DEBUG requested but host={host} is not loopback — disabling debug.")
        debug = False
    print("Starting Multi-Agent Planner Web Interface...")
    print(f"Open http://localhost:{port} in your browser")
    app.run(debug=debug, host=host, port=port)
