# server.py
import os
import uuid
import datetime
import subprocess
from pathlib import Path
import sys

from fastapi.responses import RedirectResponse, Response
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google.cloud import storage

# ------------------------- Paths & Config -------------------------

# Resolve directories based on this file's location:
#   .../Imagicle/backend/app/server.py
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

# Vendored point-e: <repo>/backend/vendor/point-e (contains top-level package folder `point_e`)
VENDORED_POINT_E_DIR = BACKEND_DIR / "vendor" / "point-e"

OUTPUT_DIR = BACKEND_DIR / "data" / "outputs" / "pointclouds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _sanitize_bucket(name: str) -> str:
    # Accept "gs://bucket" or "bucket" env values; normalize to "bucket"
    return name.replace("gs://", "").strip().strip("/")

BUCKET = _sanitize_bucket(os.getenv("POINTCLOUD_BUCKET", "imagicle-473400-pointclouds-env"))

# If you ever want to invoke the file directly instead of -m:
_env_point_e = os.getenv("POINT_E_SCRIPT")
if _env_point_e:
    _p = Path(_env_point_e)
    POINT_E_SCRIPT = _p if _p.is_absolute() else (REPO_ROOT / _p)
else:
    POINT_E_SCRIPT = BACKEND_DIR / "vendor/point-e/point_e/evals/scripts/generate.py"

# Where to store optional local copies of generated PLYs for dev viewing
_env_artifacts = os.getenv("ARTIFACTS_DIR", str(BACKEND_DIR / "artifacts"))
ARTIFACTS_DIR = Path(_env_artifacts)
if not ARTIFACTS_DIR.is_absolute():
    ARTIFACTS_DIR = BACKEND_DIR / ARTIFACTS_DIR
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Frontend origins allowed to call THIS FastAPI (separate from GCS bucket CORS)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173"
).split(",") if o.strip()]

# ------------------------- App & Middleware -------------------------

app = FastAPI(title="imagicle API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # allow OPTIONS preflight too
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def index():
    # Visiting http://localhost:8000/ sends you to the Swagger docs
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Silence the 404 spam for /favicon.ico during dev
    return Response(status_code=204)

# Serve local dev artifacts (optional)
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")

# ------------------------- Models -------------------------

class GenerateReq(BaseModel):
    prompt: str
    user_id: str | None = None
    # Optional tuning knobs you may expose later:
    guidance: float | None = None
    seed: int | None = None
    no_upsample: bool | None = None

# ------------------------- Routes -------------------------

@app.get("/healthz")
def health():
    return {
        "ok": True,
        "service": "imagicle",
        "bucket": BUCKET,
        "point_e_script": str(POINT_E_SCRIPT),
        "vendored_point_e_dir": str(VENDORED_POINT_E_DIR),
        "artifacts_dir": str(ARTIFACTS_DIR),
    }

# ...existing imports and config...

@app.post("/api/generate")
def generate_pointcloud(req: GenerateReq):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    job_id = str(uuid.uuid4())[:8]
    user = (req.user_id or "anon").replace("/", "_")
    #local_out = OUTPUT_DIR / f"{job_id}.ply"

    # Build the command, passing through optional knobs
    #cmd = [sys.executable, "-m", "point_e.evals.scripts.generate"]
    #cmd += ["--prompt", req.prompt]
    #if req.guidance is not None:
       # cmd += ["--guidance", str(req.guidance)]
    #if req.seed is not None:
        #cmd += ["--seed", str(req.seed)]
    #if req.no_upsample:
        #cmd += ["--no_upsample"]

    # Simplify the command - no need for arguments anymore
    cmd = [sys.executable, str(POINT_E_SCRIPT)]
    
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{VENDORED_POINT_E_DIR}:{os.environ.get('PYTHONPATH','')}",
        "POINT_E_PROMPT": req.prompt,  # Add this line
        "POINT_E_OUT": str(OUTPUT_DIR)  # Add this line too
    }

    proc = subprocess.run(
        cmd,
        cwd=str(OUTPUT_DIR),
        capture_output=True,
        text=True,
        env=env,
    )

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "Point-E failed")[:4000]
        raise HTTPException(status_code=500, detail=f"Point-E error:\n{msg}")
    #if not local_out.exists() or local_out.stat().st_size == 0:
        #msg = (proc.stderr or proc.stdout or "Output not found or empty")[:4000]
        #raise HTTPException(status_code=500, detail=f"Point-E output missing:\n{msg}")
    
    ply_files = list(OUTPUT_DIR.glob("*.ply"))
    if not ply_files:
        msg = (proc.stderr or proc.stdout or "No PLY file generated")[:4000]
        raise HTTPException(status_code=500, detail=f"Point-E output missing:\n{msg}")
    
    local_out = max(ply_files, key=lambda p: p.stat().st_mtime)
    
    if local_out.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="Generated PLY file is empty")
    
    job_specific_file = OUTPUT_DIR / f"{job_id}.ply"
    local_out.rename(job_specific_file)
    local_out = job_specific_file

    object_path = f"pointclouds/{user}/{job_id}/output.ply"
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET)
        blob = bucket.blob(object_path)
        blob.content_type = "application/octet-stream"
        blob.cache_control = "public, max-age=86400"
        blob.upload_from_filename(str(local_out))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GCS upload failed: {e}")

    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=15),
        method="GET",
        response_disposition='inline; filename="output.ply"',
    )

    # Save a local dev artifact (optional)
    try:
        (ARTIFACTS_DIR / f"{job_id}.ply").write_bytes(local_out.read_bytes())
    except Exception:
        pass  # non-fatal

    return {
        "job_id": job_id,
        "gcs_uri": f"gs://{BUCKET}/{object_path}",
        "url": url,  # FE should fetch this
    }

# ...existing code...
@app.get("/api/pointcloud/url")
def sign_existing(object_path: str = Query(..., description="e.g. pointclouds/anon/<job>/output.ply")):
    """Sign an existing object path and return a temporary GET URL."""
    client = storage.Client()
    blob = client.bucket(BUCKET).blob(object_path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="Object not found")
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=15),
        method="GET",
        response_disposition='inline; filename="output.ply"',
    )
    return {"url": url}
