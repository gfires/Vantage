"""
app.py — WhiteLights Streamlit UI

Upload → Analyze → Annotated video + rep summary table.
"""

import tempfile
import threading
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from depth_detector import _ensure_model, _get_rotation
import visualize as _viz_module
from visualize import _analyze, _render, _rep_warnings

# Disable GUI window and auto-open — headless Streamlit environment
_viz_module.SHOW_LIVE = False
_viz_module.SAVE_VIDEO = True  # still write the file, just don't open it

# Suppress subprocess.run("open ...") that fires after render
import subprocess as _subprocess
import types as _types
_noop_subprocess = _types.SimpleNamespace(run=lambda *a, **kw: None)
_viz_module.subprocess = _noop_subprocess
from params import HOLE_EXIT_FRACTION

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WhiteLights",
    page_icon="⬜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stApp"] {
    background-color: #0d0d0f;
    color: #f0f0f5;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Hero header ── */
.wl-hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
}
.wl-wordmark {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f0f0f5;
    line-height: 1;
}
.wl-wordmark span {
    color: #2563eb;
}
.wl-tagline {
    margin-top: 0.5rem;
    font-size: 0.95rem;
    color: #6b7280;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 1px solid #2a2a30 !important;
    border-radius: 10px;
    background: #1a1a1f !important;
    padding: 1rem;
}
[data-testid="stFileUploader"] label {
    color: #6b7280 !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background-color: #2563eb !important;
}
[data-testid="stProgress"] {
    background: #1a1a1f !important;
    border-radius: 4px;
}

/* ── Section label ── */
.wl-section {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

/* ── Rep table ── */
.wl-table-wrap {
    background: #1a1a1f;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    overflow-x: auto;
}
.wl-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
    color: #f0f0f5;
}
.wl-table th {
    text-align: center;
    color: #6b7280;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0 0.75rem 0.75rem;
    border-bottom: 1px solid #2a2a30;
}
.wl-table th.row-label { text-align: left; }
.wl-table td {
    text-align: center;
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid #16161a;
    color: #d1d5db;
}
.wl-table td.row-label {
    text-align: left;
    color: #6b7280;
    font-size: 0.8rem;
    font-weight: 500;
    white-space: nowrap;
}
.wl-table tr:last-child td { border-bottom: none; }
.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
}
.badge-pass { background: rgba(34,197,94,0.15); color: #22c55e; }
.badge-fail { background: rgba(239,68,68,0.15); color: #ef4444; }
.badge-borderline { background: rgba(245,158,11,0.15); color: #f59e0b; }
.warn-pill {
    display: inline-block;
    background: rgba(239,68,68,0.1);
    color: #fca5a5;
    border-radius: 3px;
    padding: 0.1rem 0.4rem;
    font-size: 0.7rem;
    font-weight: 600;
    margin: 1px 2px;
    white-space: nowrap;
}

/* ── Video player ── */
[data-testid="stVideo"] video {
    border-radius: 8px;
    width: 100%;
}

/* ── Divider ── */
hr { border-color: #1e1e24; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="wl-hero">
    <div class="wl-wordmark">White<span>Lights</span></div>
    <div class="wl-tagline">IPF depth analysis · low-bar squat</div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────

col_upload = st.columns([1, 2, 1])[1]
with col_upload:
    uploaded = st.file_uploader(
        "Upload a squat video",
        type=["mp4", "mov", "MOV", "MP4"],
        label_visibility="collapsed",
    )

if uploaded is None:
    st.markdown(
        "<p style='text-align:center; color:#6b7280; margin-top:0.5rem;"
        " font-size:0.85rem;'>Accepts MP4 or MOV · side-profile angle</p>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Processing ────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)

# Save upload to temp file
suffix = Path(uploaded.name).suffix or ".mp4"
tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
tmp.write(uploaded.read())
tmp.flush()
tmp.close()
input_path = tmp.name

# Probe video
cap = cv2.VideoCapture(input_path)
cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
rotation = _get_rotation(cap)

_ensure_model()

status_text = st.empty()
bar = st.progress(0)

# Phase 1 — analyze in a thread; drive progress bar over 2/3 * video duration
duration_s = total_frames / fps
analyze_budget_s = (2 / 3) * duration_s

analysis_result = [None]
analysis_done = threading.Event()

def _run_analyze():
    analysis_result[0] = _analyze(cap, rotation, fps, force_side=None)
    cap.release()
    analysis_done.set()

status_text.markdown(
    "<p style='text-align:center;color:#6b7280;font-size:0.85rem;'>"
    "Analyzing&hellip;</p>",
    unsafe_allow_html=True,
)

t = threading.Thread(target=_run_analyze, daemon=True)
t.start()

TICK = 0.1
elapsed = 0.0
while not analysis_done.is_set():
    pct = min(int(elapsed / analyze_budget_s * 100), 99)
    bar.progress(pct)
    time.sleep(TICK)
    elapsed += TICK

# Analysis finished — clear bar immediately, show Rendering
bar.empty()
status_text.markdown(
    "<p style='text-align:center;color:#6b7280;font-size:0.85rem;'>"
    "Rendering&hellip;</p>",
    unsafe_allow_html=True,
)

analysis = analysis_result[0]

if analysis is None:
    status_text.empty()
    st.error(
        "Could not detect a squat in this video. "
        "Make sure it's a clear side-profile shot."
    )
    st.stop()

frames_data, draw_frames, side, reps, smooth_hip_ys, knee_ys, valid_frame_indices = analysis

# Phase 2 — render annotated video
output_path = input_path.rsplit(".", 1)[0] + "_annotated.mp4"
cap2 = cv2.VideoCapture(input_path)
cap2.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
_render(
    cap2, rotation, fps,
    frames_data, draw_frames, side, reps,
    smooth_hip_ys, knee_ys, valid_frame_indices,
    output_path,
)
cap2.release()

status_text.empty()

# ── Results ───────────────────────────────────────────────────────────────────

vid_col, tbl_col = st.columns([55, 45])

with vid_col:
    st.markdown("<div class='wl-section'>Annotated Playback</div>", unsafe_allow_html=True)
    with open(output_path, "rb") as f:
        st.video(f.read())

with tbl_col:
    st.markdown("<div class='wl-section'>Rep Summary</div>", unsafe_allow_html=True)

    # Build table rows
    def _hole_s(rep) -> str:
        t = rep.get("tempo", {})
        bottom = rep["bottom_global"]
        end = rep["end_global"]
        asc_total = max(end - bottom, 1)
        hole_frames = max(1, int(asc_total * HOLE_EXIT_FRACTION))
        fps_approx = asc_total / max(t.get("ascent_s", 1) or 1, 1e-6)
        return f"{hole_frames / fps_approx:.2f}s"

    def _result_badge(result: str) -> str:
        cls = {"pass": "badge-pass", "fail": "badge-fail"}.get(
            result.lower(), "badge-borderline"
        )
        return f'<span class="badge {cls}">{result.upper()}</span>'

    def _warn_html(rep) -> str:
        raw = _rep_warnings(rep)
        if raw == "--":
            return '<span style="color:#374151;">—</span>'
        pills = "".join(
            f'<span class="warn-pill">{w.strip()}</span>'
            for w in raw.split(",") if w.strip()
        )
        return pills

    n = len(reps)
    rep_headers = [f"Rep {i}" for i in range(1, n + 1)]

    rows = [
        ("Result",       [_result_badge(r["result"])                                               for r in reps]),
        ("Descent",      [f"{r['tempo'].get('descent_s', 0):.2f}s"                                 for r in reps]),
        ("Hole",         [_hole_s(r)                                                                for r in reps]),
        ("Ascent",       [f"{r['tempo'].get('ascent_s', 0):.2f}s"                                  for r in reps]),
        ("Depth angle",  [(f"{r['depth_angle']:+.1f}°" if r.get("depth_angle") is not None else "—") for r in reps]),
        ("Max shin",     [(f"{r['tibial']['max_angle']:.0f}°"  if r["tibial"].get("max_angle") is not None else "—") for r in reps]),
        ("Warnings",     [_warn_html(r)                                                             for r in reps]),
    ]

    # Build HTML table
    header_cells = '<th class="row-label"></th>' + "".join(
        f"<th>{h}</th>" for h in rep_headers
    )
    body = ""
    for label, cells in rows:
        tds = "".join(f"<td>{c}</td>" for c in cells)
        body += f'<tr><td class="row-label">{label}</td>{tds}</tr>'

    table_html = f"""
    <div class="wl-table-wrap">
      <table class="wl-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # Rep count summary line
    n_pass = sum(1 for r in reps if r["result"] == "pass")
    n_fail = sum(1 for r in reps if r["result"] == "fail")
    n_border = n - n_pass - n_fail
    summary_parts = []
    if n_pass:
        summary_parts.append(f'<span style="color:#22c55e;">{n_pass} pass</span>')
    if n_fail:
        summary_parts.append(f'<span style="color:#ef4444;">{n_fail} fail</span>')
    if n_border:
        summary_parts.append(f'<span style="color:#f59e0b;">{n_border} borderline</span>')
    st.markdown(
        f"<p style='font-size:0.8rem;color:#6b7280;margin-top:0.75rem;'>"
        f"{n} rep{'s' if n != 1 else ''} detected &nbsp;·&nbsp; "
        + " &nbsp;·&nbsp; ".join(summary_parts)
        + f" &nbsp;·&nbsp; {side} side</p>",
        unsafe_allow_html=True,
    )
