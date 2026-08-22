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
from typing import Optional, Tuple

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
            "Color": r.color.capitalize() if r.color and r.color.lower() != "unknown" else (r.color or "—"),
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

def process_video(
    uploaded_video,
    local_video_path: str = "",
    fast_mode: bool = False,
    progress=gr.Progress(),
) -> Tuple[Optional[str], str, pd.DataFrame]:
    """
    Processes a video file (either uploaded via browser or specified by local file path)
    frame-by-frame and exports an annotated output video.
    """
    video_path = local_video_path.strip() if local_video_path and local_video_path.strip() else uploaded_video

    if not video_path:
        return None, "No video provided. Upload a file or enter a local path.", records_to_df([])

    if hasattr(video_path, "name"):
        video_path = video_path.name
    elif isinstance(video_path, dict):
        video_path = video_path.get("name") or video_path.get("path") or str(video_path)
    video_path = str(video_path).strip('"\'')

    try:
        reset_pipeline()
        _all_records.clear()
        pipeline = get_pipeline()

        # ── Temporarily disable OCR in fast mode for maximum speed ────────
        if fast_mode:
            pipeline.ocr_every_n = 999999   # effectively disables OCR

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "❌ Could not open video file. Please check the file path or format.", records_to_df([])

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 0:
            total_frames = 0

        # If width or height could not be queried, read a probe frame
        if width <= 0 or height <= 0:
            ret, probe = cap.read()
            if not ret or probe is None:
                cap.release()
                return None, "❌ Could not read any video frames.", records_to_df([])
            height, width = probe.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        duration_sec = (total_frames / max(fps, 1.0)) if total_frames > 0 else 0.0

        logger.info(f"Video: {width}×{height} @ {fps:.1f}fps, {total_frames} frames ({duration_sec:.1f}s)")

        # ── Output dimensions (preserve native 4K / 1080p resolution) ────
        out_w, out_h = width, height
        # Ensure dimensions are even (FFmpeg requirement)
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1
        logger.info(f"Encoding output at resolution {out_w}×{out_h} (CRF 18 High Quality)")

        # ── Frame stride: stride=1 processes every frame for seamless Kalman tracking
        stride = 1
        output_fps = fps

        # ── H.264 video writer (high quality CRF 18 to preserve crystal-clear text) ──
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
            quality=9,
            macro_block_size=1,
            output_params=["-crf", "18", "-preset", "fast"],
        )
        ffmpeg_writer.send(None)  # initialize stream

        frame_idx = 0
        try:
            # Handle both known and unknown frame counts (streams / webm)
            if total_frames > 0:
                frame_iterator = progress.tqdm(range(total_frames), desc="Processing frames")
            else:
                frame_iterator = iter(int, 1)  # infinite generator until EOF

            for idx in frame_iterator:
                if total_frames <= 0 or idx % stride == 0:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    # Process native full-resolution frame
                    annotated, _ = pipeline.process_frame(frame)
                    # Guarantee annotated frame matches FFmpeg buffer dimensions
                    if annotated.shape[1] != out_w or annotated.shape[0] != out_h:
                        annotated = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)
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

        if frame_idx == 0:
            return None, "❌ Video contained no readable frames.", records_to_df([])

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

def process_image(image):
    if image is None:
        return None, "⚠️ Please upload an image first."

    # Handle Gradio Image dict format (e.g. if editor/brush enabled)
    if isinstance(image, dict):
        image = image.get("image") or image.get("composite")
        if image is None:
            return None, "⚠️ Please upload an image first."

    # Handle PIL Image
    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))

    if not isinstance(image, np.ndarray) or getattr(image, "size", 0) == 0:
        return None, "⚠️ Invalid image format."

    pipeline = get_pipeline()
    try:
        # Gradio passes RGB numpy arrays — convert to BGR for OpenCV
        if len(image.shape) == 2:
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return None, "❌ Unsupported image channels."

        annotated_bgr, records = pipeline.process_image(frame_bgr)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        if not records:
            return annotated_rgb, "ℹ️ No vehicles detected in this image."

        lines = [f"### 🚗 {len(records)} Vehicle(s) Detected\n"]
        for idx, r in enumerate(records, 1):
            color = f"{r.color.capitalize()} ({r.color_conf:.0%})" if r.color and r.color.lower() != "unknown" else "Unknown"
            v_type = f"{r.vehicle_type} ({r.type_conf:.0%})" if r.vehicle_type and r.vehicle_type != "Unknown" else "Unknown"
            plate = f"`{r.plate_text}` ({r.plate_conf:.0%})" if r.plate_text else "None detected"
            lines.append(f"**Vehicle #{idx}** | **Color:** {color} | **Type:** {v_type} | **Plate:** {plate}")

        return annotated_rgb, "\n\n".join(lines)
    except Exception as e:
        logger.exception("Error during image processing:")
        return (image if isinstance(image, np.ndarray) else None), f"❌ Error processing image: {str(e)}"


# ── Tab 3: Live Webcam ──────────────────────────────────────────────────────

def webcam_frame(frame: np.ndarray):
    """Gradio streaming callback — called once per webcam frame."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return frame

    pipeline = get_pipeline()
    try:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        annotated_bgr, _ = pipeline.process_frame(frame_bgr)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb
    except Exception as e:
        logger.error(f"Error in webcam frame callback: {e}")
        return frame


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
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.patch.set_facecolor("#1E1E2E")
        ax1.set_facecolor("#1E1E2E")
        ax1.text(0.5, 0.5, "No data yet — run Video Analysis first",
                ha="center", va="center", fontsize=12, color="#888")
        ax1.axis("off")

        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.patch.set_facecolor("#1E1E2E")
        ax2.set_facecolor("#1E1E2E")
        ax2.text(0.5, 0.5, "No data yet — run Video Analysis first",
                ha="center", va="center", fontsize=12, color="#888")
        ax2.axis("off")
        return fig1, fig2

    df = records_to_df(_all_records)

    # ── Color chart ─────────────────────────────────────────────────────
    color_counts = df[df["Color"] != "Unknown"]["Color"].value_counts()
    COLOR_MAP = {
        "Beige": "#D4C4A8", "Black": "#222222", "Blue": "#3A6FD8", "Brown": "#7B4A2D",
        "Gold": "#D4AF37", "Green": "#2E9E44", "Grey": "#888888", "Gray": "#888888",
        "Orange": "#F47820", "Pink": "#FF69B4", "Purple": "#800080", "Red": "#D63B3B",
        "Silver": "#BBBFC4", "Tan": "#D2B48C", "White": "#EBEBEB", "Yellow": "#F5D200",
    }
    bar_colors = [COLOR_MAP.get(c, "#666666") for c in color_counts.index]

    fig_color, ax = plt.subplots(figsize=(7, 4))
    fig_color.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#1E1E2E")
    if not color_counts.empty:
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
    else:
        ax.text(0.5, 0.5, "No identified vehicle colors yet", ha="center", va="center", color="#888", fontsize=11)
        ax.axis("off")
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
    else:
        ax2.text(0.5, 0.5, "No identified vehicle body types yet", ha="center", va="center", color="#888", fontsize=11)
        ax2.axis("off")
    plt.tight_layout()

    return fig_color, fig_type


# ── Tab 5: Export log ───────────────────────────────────────────────────────

def export_log_csv():
    if not _all_records:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    path = tmp.name
    tmp.close()
    export_csv(_all_records, path)
    return path


def export_log_json():
    if not _all_records:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    path = tmp.name
    tmp.close()
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
                        label="Option A: Upload Video File",
                        height=180,
                        sources=["upload"],
                    )
                    local_path_input = gr.Textbox(
                        label="Option B: Local Video Path (Bypasses web upload limit)",
                        placeholder=r"e.g. C:\Code Projects\Trafic Videos\clip_005.mp4",
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
                inputs=[vid_input, local_path_input, fast_mode_chk],
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
        server_name="0.0.0.0",
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
