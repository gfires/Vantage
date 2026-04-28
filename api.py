"""
api.py — WhiteLights FastAPI backend

Endpoints:
  GET  /                  → serve index.html
  POST /upload            → save video, start analysis, return job_id
  GET  /stream/{job_id}   → MJPEG stream of annotated frames as they render
  GET  /status/{job_id}   → job state + rep data once done
  GET  /download/{job_id} → serve final H.264 MP4
"""

import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from depth_detector import _ensure_model, _get_rotation
from params import HOLE_EXIT_FRACTION, TIBIAL_NOTE_DEG, TIBIAL_WARN_DEG
from visualize import _analyze, _render

app = FastAPI()  # main application instance

# ── Job store ─────────────────────────────────────────────────────────────────

jobs: dict[str, dict] = {}


def _new_job() -> tuple[str, dict]:
    job_id = str(uuid.uuid4())
    job = {
        "state": "analyzing",
        "queue": queue.Queue(),
        "reps": None,
        "fps": None,
        "duration_s": None,
        "output_path": None,
        "error": None,
    }
    jobs[job_id] = job
    return job_id, job


# ── Rep serialisation (mirrors _output_rep_table rows) ───────────────────────

def _rep_warnings(rep: dict) -> list[str]:
    warns = list(rep["tempo"].get("flags", []))
    max_tib = rep["tibial"].get("max_angle")
    if max_tib is not None:
        if max_tib > TIBIAL_WARN_DEG:
            warns.append("KNEES TOO FORWARD")
        elif max_tib > TIBIAL_NOTE_DEG:
            warns.append("KNEES SLIGHTLY FORWARD")
    return warns


def _hole_s(rep: dict) -> float:
    t = rep.get("tempo", {})
    bottom = rep["bottom_global"]
    end = rep["end_global"]
    asc_total = max(end - bottom, 1)
    hole_frames = max(1, int(asc_total * HOLE_EXIT_FRACTION))
    fps_approx = asc_total / max(t.get("ascent_s", 1) or 1, 1e-6)
    return round(hole_frames / fps_approx, 2)


def _serialise_reps(reps: list) -> list[dict]:
    out = []
    for i, rep in enumerate(reps, 1):
        t = rep["tempo"]
        tib = rep["tibial"]
        out.append({
            "rep": i,
            "result": rep["result"].upper(),
            "descent_s": round(t.get("descent_s", 0), 2),
            "hole_s": _hole_s(rep),
            "ascent_s": round(t.get("ascent_s", 0), 2),
            "depth_angle": round(rep["depth_angle"], 1) if rep.get("depth_angle") is not None else None,
            "max_shin": round(tib["max_angle"], 0) if tib.get("max_angle") is not None else None,
            "warnings": _rep_warnings(rep),
        })
    return out


# ── Background worker ─────────────────────────────────────────────────────────

def _process(job_id: str, input_path: str, output_path: str) -> None:
    job = jobs[job_id]
    try:
        _ensure_model()

        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        rotation = _get_rotation(cap)
        job["fps"] = fps

        analysis = _analyze(cap, rotation, fps, force_side="right")
        cap.release()

        if analysis is None:
            job["state"] = "error"
            job["error"] = "Could not detect a squat. Check camera angle."
            job["queue"].put(None)
            return

        frames_data, draw_frames, side, reps, smooth_hip_ys, knee_ys, valid_frame_indices = analysis
        job["reps"] = _serialise_reps(reps)
        job["state"] = "rendering"

        frame_interval = 1.0 / fps
        _last_frame_time: list[float] = [time.monotonic()]

        def _cb(data: bytes | None) -> None:
            if data is not None:
                now = time.monotonic()
                elapsed = now - _last_frame_time[0]
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                _last_frame_time[0] = time.monotonic()
            job["queue"].put(data)

        cap2 = cv2.VideoCapture(input_path)
        cap2.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        _render(
            cap2, rotation, fps,
            frames_data, draw_frames, side, reps,
            smooth_hip_ys, knee_ys, valid_frame_indices,
            output_path,
            frame_callback=_cb,
        )
        cap2.release()

        job["output_path"] = output_path
        job["state"] = "done"

    except Exception as exc:
        job["state"] = "error"
        job["error"] = str(exc)
        job["queue"].put(None)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/upload")
async def upload(file: UploadFile):
    suffix = Path(file.filename).suffix or ".mp4"
    tmp_dir = tempfile.mkdtemp()
    input_path = str(Path(tmp_dir) / f"input{suffix}")
    output_path = str(Path(tmp_dir) / "annotated.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Probe duration from file header — instantaneous, no decoding
    _cap = cv2.VideoCapture(input_path)
    _cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    _fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
    _frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    _cap.release()
    duration_s = _frames / _fps

    job_id, job = _new_job()
    job["duration_s"] = duration_s
    threading.Thread(
        target=_process,
        args=(job_id, input_path, output_path),
        daemon=True,
    ).start()

    return {"job_id": job_id, "duration_s": duration_s}


@app.get("/stream/{job_id}")
def stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    def _generate():
        while True:
            data = job["queue"].get()
            if data is None:
                break
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + data +
                b"\r\n"
            )

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "state": job["state"],
        "reps": job["reps"],
        "error": job["error"],
    }


@app.get("/download/{job_id}")
def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["state"] != "done" or not job["output_path"]:
        raise HTTPException(status_code=404, detail="Not ready")
    return FileResponse(
        job["output_path"],
        media_type="video/mp4",
        filename="whitelights_annotated.mp4",
    )
