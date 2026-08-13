"""
dashboard.py
------------
Traffic Tracker — Gradio Web Dashboard

Launch:
    python app/dashboard.py
"""

import os
import sys

# ── Must be set BEFORE torch/numpy/cv2 are imported ────────────────────────────
# Prevents crash when PyTorch + OpenCV both link libiomp5md.dll (Windows only)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_tracker.pipeline import TrafficPipeline
from traffic_tracker.utils import VehicleRecord, export_csv, export_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config.yaml")

# ── Global pipeline (lazy init) ─────────────────────────────────────────────
_pipeline: TrafficPipeline = None


def get_pipeline() -> TrafficPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising TrafficPipeline…")
        _pipeline = TrafficPipeline(config_path=CONFIG_PATH)
    return _pipeline


def reset_pipeline():
    global _pipeline
    if _pipeline is not None:
        _pipeline.reset()


# ── Shared log state ────────────────────────────────────────────────────────
_all_records: list[VehicleRecord] = []


def records_to_df(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=[
            "Track ID", "Plate", "Plate Conf", "Color", "Color Conf",
            "Type", "Type Conf", "First Frame", "Last Frame", "Time",
        ])
    rows = []
    for r in records:
        rows.append({
            "Track ID": r.track_id,
            "Plate": r.plate_text or "—",
            "Plate Conf": f"{r.plate_conf:.0%}",
            "Color": r.color,
            "Color Conf": f"{r.color_conf:.0%}",
            "Type": r.vehicle_type,
            "Type Conf": f"{r.type_conf:.0%}",
            "First Frame": r.frame_first_seen,
            "Last Frame": r.frame_last_seen,
            "Time": r.timestamp,
        })
    return pd.DataFrame(rows)



def _reencode_h264(src_path: str) -> str:
    """
    Re-encode a mp4v video to H.264 so it's playable in browsers / Gradio.
    Uses imageio-ffmpeg which ships with Gradio. Falls back to src_path on error.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        dst_path = src_path.replace(".mp4", "_h264.mp4")
        import subprocess
        result = subprocess.run(
            [
                ffmpeg_exe, "-y",
                "-i", src_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",   # enables streaming / instant play
                "-an",                        # no audio track needed
                dst_path,
            ],
            capture_output=True,
            timeout=600,
        )
        if result.returncode == 0:
            import os as _os
            _os.unlink(src_path)   # clean up mp4v file
            logger.info(f"Re-encoded to H.264: {dst_path}")
            return dst_path
        else:
            logger.warning(f"ffmpeg re-encode failed: {result.stderr.decode()[:200]}")
    except Exception as e:
        logger.warning(f"H.264 re-encode skipped ({e}); serving mp4v directly")
    return src_path   # serve as-is if re-encode fails



# ── Tab 1: Video Analysis ───────────────────────────────────────────────────

def process_video(video_path, fast_mode=False, progress=gr.Progress(track_tqdm=True)):
    if not video_path:
        return None, "No video uploaded.", records_to_df([])

    # gr.Video returns a plain string path; normalise just in case
    if hasattr(video_path, "name"):
        video_path = video_path.name
    elif isinstance(video_path, dict):
        video_path = video_path.get("name") or video_path.get("path") or str(video_path)
    video_path = str(video_path)

    try:
        reset_pipeline()
        _all_records.clear()
        pipeline = get_pipeline()

        # ── Temporarily disable OCR in fast mode for maximum speed ────────
        if fast_mode:
            pipeline.ocr_every_n = 999999   # effectively disables OCR

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "❌ Could not open video file.", records_to_df([])

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / max(fps, 1)

        logger.info(f"Video: {width}×{height} @ {fps:.1f}fps, {total_frames} frames ({duration_sec:.1f}s)")

        # ── Downscale 4K output to 1080p for browser playback & fast processing
        out_w, out_h = width, height
        if max(width, height) > 1920:
            scale_out = 1920 / max(width, height)
            out_w = int(width * scale_out)
            out_h = int(height * scale_out)
            # Make width and height even (FFmpeg requirement)
            out_w = out_w if out_w % 2 == 0 else out_w - 1
            out_h = out_h if out_h % 2 == 0 else out_h - 1
            logger.info(f"Downscaling output {width}×{height} → {out_w}×{out_h}")

        # ── Frame stride: stride=2 processes every 2nd frame (optimal for RTX 4080)
        stride = 2 if fps >= 25 else 1
        output_fps = fps / stride

        # ── H.264 video writer (stream directly via imageio_ffmpeg) ───────
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_path = tmp.name
        tmp.close()

        import imageio_ffmpeg
        ffmpeg_writer = imageio_ffmpeg.write_frames(
            out_path,
            (out_w, out_h),
            fps=output_fps,
            codec="libx264",
            pix_fmt_in="bgr24",
            quality=6,
            macro_block_size=1,
        )
        ffmpeg_writer.send(None)  # initialize stream

        frame_idx = 0
        try:
            for idx in progress.tqdm(range(total_frames), desc="Processing frames"):
                if idx % stride == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Downscale frame before processing if 4K
                    if (out_w, out_h) != (width, height):
                        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                    annotated, _ = pipeline.process_frame(frame)
                    ffmpeg_writer.send(annotated)
                    frame_idx += 1
                else:
                    # Fast skip frame without decoding image pixels (100x faster than set/seek)
                    ret = cap.grab()
                    if not ret:
                        break
        finally:
            cap.release()
            try:
                ffmpeg_writer.close()
            except Exception:
                pass

        _all_records.extend(pipeline.all_records)

        mode_label = " (fast mode — OCR skipped)" if fast_mode else ""
        status = (
            f"✅ Processed {frame_idx} frames{mode_label} — "
            f"detected {len(_all_records)} unique vehicles"
        )
        return out_path, status, records_to_df(_all_records)

    except Exception as e:
        logger.exception("Error during video processing:")
        return None, f"❌ Processing Error: {str(e)}", records_to_df([])


# ── Tab 2: Image Analysis ───────────────────────────────────────────────────

def process_image(image: np.ndarray):
    if image is None:
        return None, "No image uploaded."

    pipeline = get_pipeline()
    # Gradio passes RGB numpy arrays — convert to BGR for OpenCV
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_bgr, records = pipeline.process_image(frame_bgr)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    lines = [f"**{len(records)} vehicle(s) detected:**"]
    for r in records:
        parts = []
        if r.color and r.color != "Unknown":
            parts.append(f"**Color:** {r.color} ({r.color_conf:.0%})")
        if r.vehicle_type and r.vehicle_type != "Unknown":
            parts.append(f"**Type:** {r.vehicle_type} ({r.type_conf:.0%})")
        if r.plate_text:
            parts.append(f"**Plate:** `{r.plate_text}`")
        if parts:
            lines.append("- " + " | ".join(parts))

    return annotated_rgb, "\n".join(lines)


# ── Tab 3: Live Webcam ──────────────────────────────────────────────────────

def webcam_frame(frame: np.ndarray):
    """Gradio streaming callback — called once per webcam frame."""
    if frame is None:
        return frame

    pipeline = get_pipeline()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_bgr, _ = pipeline.process_frame(frame_bgr)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb


# ── Tab 4: Analytics ────────────────────────────────────────────────────────

def build_charts():
    """Build matplotlib charts from the current detection log."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None

    if not _all_records:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data yet — run Video Analysis first",
                ha="center", va="center", fontsize=13, color="#888")
        ax.axis("off")
        return fig, fig

    df = records_to_df(_all_records)

    # ── Color chart ─────────────────────────────────────────────────────
    color_counts = df[df["Color"] != "Unknown"]["Color"].value_counts()
    COLOR_MAP = {
        "Black": "#222", "Blue": "#3A6FD8", "Brown": "#7B4A2D",
        "Green": "#2E9E44", "Gray": "#888", "Orange": "#F47820",
        "Red": "#D63B3B", "Silver": "#BBBFC4", "White": "#EBEBEB", "Yellow": "#F5D200",
    }
    bar_colors = [COLOR_MAP.get(c, "#666") for c in color_counts.index]

    fig_color, ax = plt.subplots(figsize=(7, 4))
    fig_color.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#1E1E2E")
    bars = ax.bar(color_counts.index, color_counts.values, color=bar_colors, edgecolor="#333", linewidth=0.8)
    ax.set_title("Vehicles by Color", color="#E0E0E0", fontsize=14, pad=12)
    ax.tick_params(colors="#AAA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#444")
    ax.yaxis.label.set_color("#AAA")
    for bar, val in zip(bars, color_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", color="#DDD", fontsize=10)
    plt.tight_layout()

    # ── Type chart ───────────────────────────────────────────────────────
    type_counts = df[df["Type"] != "Unknown"]["Type"].value_counts()
    TYPE_COLORS = ["#4C9BE8", "#6BD48B", "#F4A261", "#E76F51",
                   "#9B5DE5", "#F5D200", "#00B4D8"]

    fig_type, ax2 = plt.subplots(figsize=(5, 5))
    fig_type.patch.set_facecolor("#1E1E2E")
    ax2.set_facecolor("#1E1E2E")
    if not type_counts.empty:
        wedges, texts, autotexts = ax2.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.0f%%",
            colors=TYPE_COLORS[:len(type_counts)],
            startangle=90,
            wedgeprops={"edgecolor": "#1E1E2E", "linewidth": 2},
        )
        for t in texts + autotexts:
            t.set_color("#DDD")
    ax2.set_title("Vehicles by Body Type", color="#E0E0E0", fontsize=14, pad=12)
    plt.tight_layout()

    return fig_color, fig_type


# ── Tab 5: Export log ───────────────────────────────────────────────────────

def export_log_csv():
    if not _all_records:
        return None
    path = tempfile.mktemp(suffix=".csv")
    export_csv(_all_records, path)
    return path


def export_log_json():
    if not _all_records:
        return None
    path = tempfile.mktemp(suffix=".json")
    export_json(_all_records, path)
    return path


# ── Gradio UI ───────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #0F0F1A !important; }
.tab-nav button { font-size: 15px !important; }
h1 { 
    background: linear-gradient(135deg, #6DD5FA, #2980B9, #9B59B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
.status-box { 
    border-radius: 8px; 
    background: #161625 !important; 
    border: 1px solid #2A2A4A !important; 
}
footer { display: none !important; }
"""

with gr.Blocks(title="Traffic Tracker") as demo:

    gr.Markdown(
        "# Traffic Tracker\n"
        "License plate recognition, vehicle color, and body type detection."
    )

    with gr.Tabs():

        # ── Video Analysis ───────────────────────────────────────────────
        with gr.Tab("Video Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(
                        label="Upload Video File (.mp4, .mov, .avi, .mkv)",
                        height=240,
                        sources=["upload"],
                        max_file_size="1000mb",
                    )
                    with gr.Row():
                        vid_btn = gr.Button("Analyse Video", variant="primary", size="lg")
                        fast_mode_chk = gr.Checkbox(
                            label="Fast Mode (skip OCR)",
                            value=False,
                            info="Detects vehicles and colors but skips plate reading",
                        )
                    vid_status = gr.Textbox(
                        label="Status", interactive=False,
                        elem_classes=["status-box"]
                    )
                with gr.Column(scale=1):
                    vid_output = gr.Video(label="Annotated Output", height=280)

            vid_table = gr.DataFrame(label="Detection Results", wrap=True)
            vid_btn.click(
                fn=process_video,
                inputs=[vid_input, fast_mode_chk],
                outputs=[vid_output, vid_status, vid_table],
            )

        # ── Image Analysis ───────────────────────────────────────────────
        with gr.Tab("Image Analysis"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="Upload Image", type="numpy", height=320)
                    img_btn = gr.Button("Analyse Image", variant="primary")
                with gr.Column():
                    img_output = gr.Image(label="Annotated Output", type="numpy", height=320)
                    img_results = gr.Markdown("*Upload an image and click Analyse.*")
            img_btn.click(
                fn=process_image,
                inputs=[img_input],
                outputs=[img_output, img_results],
            )

        # ── Live Webcam ──────────────────────────────────────────────────
        with gr.Tab("Live Webcam"):
            gr.Markdown(
                "> Click **Start** to begin webcam detection.  \n"
                "> Detection runs every frame."
            )
            webcam = gr.Image(
                label="Webcam Feed",
                sources=["webcam"],
                streaming=True,
                type="numpy",
                height=420,
            )
            webcam_out = gr.Image(label="Annotated Feed", type="numpy", height=420)
            webcam.stream(fn=webcam_frame, inputs=[webcam], outputs=[webcam_out])

        # ── Analytics ────────────────────────────────────────────────────
        with gr.Tab("Analytics"):
            chart_btn = gr.Button("Refresh Charts", variant="secondary")
            with gr.Row():
                color_chart = gr.Plot(label="Vehicles by Colour")
                type_chart = gr.Plot(label="Vehicles by Body Type")
            chart_btn.click(
                fn=build_charts,
                inputs=[],
                outputs=[color_chart, type_chart],
            )

        # ── Detection Log ────────────────────────────────────────────────
        with gr.Tab("Detection Log"):
            gr.Markdown(
                "All vehicles detected across analysed videos. "
                "Download the log as CSV or JSON."
            )
            log_table = gr.DataFrame(
                value=records_to_df([]),
                label="Vehicle Log",
                wrap=True,
            )
            with gr.Row():
                csv_btn = gr.Button("Export CSV")
                json_btn = gr.Button("Export JSON")
                csv_dl = gr.File(label="CSV Download")
                json_dl = gr.File(label="JSON Download")

            csv_btn.click(fn=export_log_csv, inputs=[], outputs=[csv_dl])
            json_btn.click(fn=export_log_json, inputs=[], outputs=[json_dl])

            # Refresh log table after video analysis
            vid_btn.click(
                fn=lambda: records_to_df(_all_records),
                inputs=[],
                outputs=[log_table],
            )

    gr.Markdown(
        "<div style='text-align:center; color:#555; margin-top:16px; font-size:12px;'>"
        "Traffic Tracker · YOLOv8 + MobileNetV3 + EasyOCR"
        "</div>"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        max_file_size="1000mb",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
        show_error=True,
        share=False,
    )
