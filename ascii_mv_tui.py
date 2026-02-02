#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from yt_dlp import YoutubeDL

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    ProgressBar,
    RichLog,
    Select,
    Static,
)
from textual.worker import get_current_worker


ASCII_RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
Mode = Literal["ascii", "braille"]


class PipelineCancelled(RuntimeError):
    pass


# -------------------------
# Helpers
# -------------------------


def get_worker_or_none():
    """Textual raises if there's no active worker (e.g. when running headless)."""
    try:
        return get_current_worker()
    except Exception:
        return None


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg first.")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rmtree_retry(p: Path, *, tries: int = 6, delay: float = 0.1) -> None:
    """Robustly remove a directory that may be concurrently touched (WSL/Windows can be finicky)."""
    if not p.exists():
        return
    last_err: Exception | None = None
    for i in range(tries):
        try:
            shutil.rmtree(p)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            last_err = e
            # Retry in case files are still being written / released.
            time.sleep(delay * (i + 1))
    if last_err is not None:
        raise last_err


def find_mono_font() -> str:
    # Keep this simple and OS-friendly. Users can swap the file if desired.
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/lucon.ttf",
        # These often exist and have a wider glyph set than Consolas.
        "C:/Windows/Fonts/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not auto-find a monospaced font. Install one (e.g. DejaVu Sans Mono) or edit find_mono_font()."
    )


def human_bytes(n: float) -> str:
    if not n:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    k = 1024.0
    i = min(int(math.log(n, k)), len(units) - 1)
    return f"{n / (k ** i):.2f}{units[i]}"


def detect_js_runtimes() -> dict[str, dict]:
    """Return yt-dlp js_runtimes config for runtimes available on PATH."""
    runtimes: dict[str, dict] = {}
    if shutil.which("deno"):
        runtimes["deno"] = {}
    if shutil.which("node"):
        runtimes["node"] = {}
    if shutil.which("bun"):
        runtimes["bun"] = {}
    # quickjs is supported by yt-dlp, but executable names vary across platforms;
    # we don't auto-enable it unless explicitly configured by the user.
    return runtimes


def font_supports_distinct_braille(font_path: str) -> bool:
    """Detect 'tofu' (all braille chars rendered identically as missing-glyph boxes)."""
    try:
        font = ImageFont.truetype(font_path, 32)
        chars = ["\u2800", "\u2801", "\u28ff"]
        imgs = []
        for ch in chars:
            img = Image.new("L", (64, 64), 0)
            d = ImageDraw.Draw(img)
            d.text((0, 0), ch, font=font, fill=255)
            imgs.append(np.asarray(img, dtype=np.int16))
        # If all 3 are identical (or nearly), braille isn't really supported.
        d01 = int(np.abs(imgs[0] - imgs[1]).sum())
        d0f = int(np.abs(imgs[0] - imgs[2]).sum())
        d1f = int(np.abs(imgs[1] - imgs[2]).sum())
        return (d01 + d0f + d1f) > 500
    except Exception:
        return False


def find_braille_font() -> str:
    # Braille patterns are U+2800..U+28FF. Many monospace fonts don't include these.
    candidates = [
        # Windows
        "C:/Windows/Fonts/seguisym.ttf",  # Segoe UI Symbol
        "C:/Windows/Fonts/SegoeUI.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
        # Linux (DejaVu Sans includes braille; DejaVu Sans Mono often does not)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p) and font_supports_distinct_braille(p):
            return p
    raise FileNotFoundError(
        "Could not auto-find a font with Braille glyphs. "
        "Install a font that supports U+2800..U+28FF (e.g. DejaVu Sans / Noto) "
        "or edit find_braille_font()."
    )


def find_font_for_mode(mode: Mode) -> str:
    return find_braille_font() if mode == "braille" else find_mono_font()


def run_process_checked(cmd: list[str], *, log_prefix: str = "") -> None:
    """Run a subprocess and allow Textual worker cancellation to terminate it."""
    worker = get_worker_or_none()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        while proc.poll() is None:
            if worker is not None and worker.is_cancelled:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise PipelineCancelled("Cancelled.")
            time.sleep(0.05)
    finally:
        # Ensure we don't leak a live process if something above throws.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()

    stderr = ""
    if proc.stderr is not None:
        try:
            stderr = proc.stderr.read() or ""
        except Exception:
            stderr = ""

    if proc.returncode != 0:
        tail = stderr.strip()
        if tail:
            raise RuntimeError(f"{log_prefix}Command failed ({proc.returncode}).\n{tail}")
        raise RuntimeError(f"{log_prefix}Command failed ({proc.returncode}).")


def ffprobe_duration_seconds(video_path: Path) -> Optional[float]:
    """Return duration in seconds (best-effort)."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nk=1:nw=1",
                str(video_path),
            ],
            text=True,
        ).strip()
        if not out:
            return None
        return float(out)
    except Exception:
        return None


def count_frame_files(frames_dir: Path) -> int:
    if not frames_dir.exists():
        return 0
    n = 0
    with os.scandir(frames_dir) as it:
        for entry in it:
            # Fast path: only count expected pattern.
            if entry.is_file() and entry.name.startswith("frame_") and entry.name.endswith(".png"):
                n += 1
    return n


def ffmpeg_supports_encoder(name: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.DEVNULL)
        return name in out
    except Exception:
        return False


# -------------------------
# Text conversion
# -------------------------


def prepare_luma(img: Image.Image) -> Image.Image:
    """Normalize contrast to preserve detail at low character resolutions."""
    # Light autocontrast keeps highlights from washing out and pulls shadows up a bit.
    return ImageOps.autocontrast(img, cutoff=1)


def floyd_steinberg_dither(arr: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Return boolean map using Floyd-Steinberg error diffusion."""
    h, w = arr.shape
    buf = arr.astype(np.float32, copy=True)
    thresh = float(threshold)
    for y in range(h):
        for x in range(w):
            old = buf[y, x]
            new = 0.0 if old < thresh else 255.0
            err = old - new
            buf[y, x] = new
            if x + 1 < w:
                buf[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < h:
                if x > 0:
                    buf[y + 1, x - 1] += err * (3.0 / 16.0)
                buf[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < w:
                    buf[y + 1, x + 1] += err * (1.0 / 16.0)
    return buf < thresh


def img_to_ascii(img_path: Path, width_chars: int) -> str:
    img = Image.open(img_path).convert("L")
    img = prepare_luma(img)
    w, h = img.size

    aspect_fix = 0.55
    height_chars = max(1, int((h / w) * width_chars * aspect_fix))

    small = img.resize((width_chars, height_chars), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.uint8)

    idx = (arr.astype(np.float32) / 255.0) * (len(ASCII_RAMP) - 1)
    idx = idx.astype(np.int32)

    # Dark -> dense
    idx = (len(ASCII_RAMP) - 1) - idx

    lines = ["".join(ASCII_RAMP[i] for i in row) for row in idx]
    return "\n".join(lines)


def img_to_braille(img_path: Path, width_chars: int, threshold: int = 140) -> str:
    img = Image.open(img_path).convert("L")
    img = prepare_luma(img)
    w, h = img.size

    # Each Braille char is 2x4 dots, so output "character aspect" differs from ASCII.
    height_chars = max(1, int((h / w) * width_chars / 2))

    target_w = width_chars * 2
    target_h = height_chars * 4
    small = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.uint8)

    on = floyd_steinberg_dither(arr, threshold=threshold)

    # Braille dot bit layout:
    # 1 4
    # 2 5
    # 3 6
    # 7 8
    dot_bits = [
        (0, 0, 0x01),
        (0, 1, 0x02),
        (0, 2, 0x04),
        (1, 0, 0x08),
        (1, 1, 0x10),
        (1, 2, 0x20),
        (0, 3, 0x40),
        (1, 3, 0x80),
    ]

    lines: list[str] = []
    for cy in range(height_chars):
        y0 = cy * 4
        row_chars: list[str] = []
        for cx in range(width_chars):
            x0 = cx * 2
            bits = 0
            for dx, dy, b in dot_bits:
                if on[y0 + dy, x0 + dx]:
                    bits |= b
            row_chars.append(chr(0x2800 + bits))
        lines.append("".join(row_chars))
    return "\n".join(lines)


def render_text_to_png(
    text: str,
    out_path: Path,
    font_path: str,
    font_size: int,
    padding: int,
) -> None:
    font = ImageFont.truetype(font_path, font_size)
    lines = text.splitlines() or [""]

    # Use font metrics for stable line spacing. Some fonts render Braille with
    # different metrics than ASCII.
    ascent, descent = font.getmetrics()
    line_h = max(1, ascent + descent)

    # Width heuristic: prefer the braille-full-cell if present; fall back to "M".
    # This prevents under-sizing when the chosen braille-capable font isn't mono.
    sample_chars = ["\u28ff", "M"]
    char_w = 0
    char_h = 0
    for ch in sample_chars:
        try:
            bbox = font.getbbox(ch)
            char_w = max(char_w, bbox[2] - bbox[0])
            char_h = max(char_h, bbox[3] - bbox[1])
        except Exception:
            continue
    char_w = max(1, int(char_w))
    # Ensure we never under-allocate vertically vs. line metrics.
    char_h = max(1, int(max(char_h, line_h)))

    img_w = padding * 2 + char_w * max(len(line) for line in lines)
    img_h = padding * 2 + line_h * len(lines)

    canvas = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    # Draw line-by-line to avoid font-dependent newline spacing quirks.
    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=(255, 255, 255))
        y += line_h
    canvas.save(out_path)


# -------------------------
# Video pipeline
# -------------------------


def ytdlp_download(url: str, out_dir: Path, post, *, quiet: bool = True) -> Path:
    """Download a URL to out_dir and return the merged mp4 path."""
    safe_mkdir(out_dir)
    outtmpl = str(out_dir / "input.%(ext)s")
    worker = get_worker_or_none()

    # Remove any previous downloads so we don't accidentally reuse them.
    for p in out_dir.glob("input.*"):
        try:
            p.unlink()
        except Exception:
            pass

    def hook(d):
        if worker is not None and worker.is_cancelled:
            raise PipelineCancelled("Cancelled.")

        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            pct = (downloaded / total * 100.0) if total else None
            speed = d.get("speed") or 0
            eta = d.get("eta")
            post(ProgressUpdate(stage="download", percent=pct))
            if pct is not None:
                msg = f"Downloading... {pct:5.1f}% | {human_bytes(speed)}/s"
                if eta:
                    msg += f" | ETA {eta}s"
                post(LogLine(msg))
        elif d.get("status") == "finished":
            post(ProgressUpdate(stage="download", percent=100.0))
            post(LogLine("Download finished, merging streams..."))

    class _Logger:
        def debug(self, msg):  # noqa: ANN001
            # Too noisy for the TUI; keep it quiet unless you want full trace.
            return

        def warning(self, msg):  # noqa: ANN001
            post(LogLine(f"yt-dlp warning: {msg}"))

        def error(self, msg):  # noqa: ANN001
            post(LogLine(f"yt-dlp error: {msg}"))

    js_runtimes = detect_js_runtimes()

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": quiet,
        "noprogress": True,
        "logger": _Logger(),
        "progress_hooks": [hook],
    }

    # If we can find a JS runtime on PATH (node/deno/bun), enable it to reduce the
    # chance of future YouTube extraction breakage.
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes
        # Allow yt-dlp to fetch the recommended EJS solver distribution when needed.
        # This improves reliability for YouTube signature/n challenge solving.
        ydl_opts["remote_components"] = ["ejs:github"]

    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    for p in sorted(out_dir.glob("input.*"), key=lambda x: x.stat().st_mtime, reverse=True):
        return p
    raise FileNotFoundError("Download finished but no input.* file found.")


def extract_frames(video_path: Path, frames_dir: Path, fps: float, post) -> None:
    safe_mkdir(frames_dir)
    post(LogLine(f"Extracting frames at {fps:.3f} fps..."))
    # We can estimate extraction progress by expected frame count from video duration.
    duration = ffprobe_duration_seconds(video_path)
    expected = None
    if duration:
        expected = max(1, int(math.ceil(duration * fps)))
        post(LogLine(f"Video duration ~{duration:.1f}s => ~{expected} frames at {fps:.3f} fps"))
    post(ProgressUpdate(stage="extract", percent=0.0 if expected else None))

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(frames_dir / "frame_%06d.png"),
    ]

    worker = get_worker_or_none()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        last_post = 0.0
        while proc.poll() is None:
            if worker is not None and worker.is_cancelled:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise PipelineCancelled("Cancelled.")

            if expected:
                now = time.time()
                if now - last_post >= 0.5:
                    last_post = now
                    count = count_frame_files(frames_dir)
                    pct = min(99.0, (count / expected) * 100.0)
                    post(ProgressUpdate(stage="extract", percent=pct))
            time.sleep(0.05)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()

    stderr = ""
    if proc.stderr is not None:
        try:
            stderr = proc.stderr.read() or ""
        except Exception:
            stderr = ""

    if proc.returncode != 0:
        tail = stderr.strip()
        if tail:
            raise RuntimeError(f"extract: Command failed ({proc.returncode}).\n{tail}")
        raise RuntimeError(f"extract: Command failed ({proc.returncode}).")

    post(LogLine("Frame extraction done."))
    post(ProgressUpdate(stage="extract", percent=100.0))


def render_frames(
    frames_dir: Path,
    render_dir: Path,
    width_chars: int,
    mode: Mode,
    braille_threshold: int,
    font_path: str,
    font_size: int,
    padding: int,
    post,
) -> None:
    safe_mkdir(render_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    total = len(frame_paths)
    if total == 0:
        raise RuntimeError("No frames extracted. Is ffmpeg installed and the video readable?")

    post(LogLine(f"Rendering {total} frames in {mode.upper()} mode..."))
    post(ProgressUpdate(stage="render", percent=0.0))

    worker = get_worker_or_none()
    for i, img_path in enumerate(frame_paths, start=1):
        if worker is not None and worker.is_cancelled:
            raise PipelineCancelled("Cancelled.")

        if mode == "braille":
            text = img_to_braille(img_path, width_chars, threshold=braille_threshold)
        else:
            text = img_to_ascii(img_path, width_chars)

        out_png = render_dir / img_path.name
        render_text_to_png(text, out_png, font_path, font_size, padding)

        if i == 1 or i % 5 == 0 or i == total:
            post(PreviewFrame(frame=i, total=total, text=text))

        post(ProgressUpdate(stage="render", percent=(i / total) * 100.0))

    post(LogLine("Rendering complete."))


def assemble_video(render_dir: Path, fps: float, out_silent: Path, post, *, encoder: str = "auto") -> None:
    post(LogLine("Encoding frames into MP4..."))
    post(ProgressUpdate(stage="encode", percent=None))

    chosen = encoder
    if chosen == "auto":
        chosen = "h264_nvenc" if ffmpeg_supports_encoder("h264_nvenc") else "libx264"

    if chosen == "h264_nvenc" and not ffmpeg_supports_encoder("h264_nvenc"):
        post(LogLine("NVENC not available in ffmpeg; falling back to libx264."))
        chosen = "libx264"

    if chosen == "h264_nvenc":
        # GPU encode (fast). Use constant quality mode.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(render_dir / "frame_%06d.png"),
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_silent),
        ]
        try:
            run_process_checked(cmd, log_prefix="encode(nvenc): ")
            post(ProgressUpdate(stage="encode", percent=100.0))
            return
        except Exception as e:
            post(LogLine(f"NVENC failed, falling back to libx264. ({e})"))
            chosen = "libx264"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(render_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        str(out_silent),
    ]
    run_process_checked(cmd, log_prefix="encode: ")
    post(ProgressUpdate(stage="encode", percent=100.0))


def mux_audio(video_silent: Path, original_video: Path, out_final: Path, post) -> None:
    post(LogLine("Muxing original audio..."))
    post(ProgressUpdate(stage="mux", percent=None))

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_silent),
        "-i",
        str(original_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_final),
    ]
    run_process_checked(cmd, log_prefix="mux: ")
    post(ProgressUpdate(stage="mux", percent=100.0))
    post(LogLine(f"Done: {out_final}"))


# -------------------------
# Textual messages
# -------------------------


class LogLine(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class ProgressUpdate(Message):
    def __init__(self, stage: str, percent: Optional[float]) -> None:
        self.stage = stage
        self.percent = percent
        super().__init__()


class PreviewFrame(Message):
    def __init__(self, frame: int, total: int, text: str) -> None:
        self.frame = frame
        self.total = total
        self.text = text
        super().__init__()


# -------------------------
# Textual App
# -------------------------


class AsciiMV(App):
    CSS = """
    Screen { layout: vertical; }
    #top { height: auto; padding: 1; }
    #row_a { height: auto; margin-top: 1; }
    #row_b { height: auto; margin-top: 1; }
    #row_c { height: auto; margin-top: 1; }
    #main { height: 1fr; }
    #left { width: 1fr; border: round $panel; padding: 1; }
    #right { width: 2fr; border: round $panel; padding: 1; }
    #preview_box { height: 1fr; }
    #preview { height: 1fr; }
    #status_row { height: auto; margin-top: 1; }

    #fps, #width, #braille_thresh { width: 10; }
    #mode { width: 26; }
    #out { width: 1fr; }
    #encoder { width: 22; }
    """

    TITLE = "ASCII Music Video (TUI)"
    SUB_TITLE = "YouTube -> frames -> ASCII/Braille -> MP4"

    BINDINGS = [
        ("ctrl+s", "start", "Start"),
        ("ctrl+c", "cancel", "Cancel"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._stage_text: str = "Idle"

    def compose(self) -> ComposeResult:
        yield Header()

        with Vertical(id="top"):
            yield Label("Source (YouTube URL or local file path):")
            yield Input(placeholder="https://www.youtube.com/watch?v=... OR C:\\video.mp4", id="source")

            # Keep the primary action visible even on narrow terminals by splitting rows.
            with Horizontal(id="row_a"):
                yield Label("FPS:")
                yield Input(value="5", id="fps", type="number")
                yield Label("Width(chars):")
                yield Input(value="200", id="width", type="number")
                yield Label("Mode:")
                yield Select(
                    options=[("ASCII ramp", "ascii"), ("Braille (ultra-dense)", "braille")],
                    value="braille",
                    id="mode",
                )
                yield Label("Braille Thresh:")
                yield Input(value="140", id="braille_thresh", type="number")

            with Horizontal(id="row_b"):
                yield Label("Output:")
                yield Input(value="ascii_final.mp4", id="out")
                yield Label("Encoder:")
                yield Select(
                    options=[("Auto", "auto"), ("CPU (libx264)", "libx264"), ("GPU (h264_nvenc)", "h264_nvenc")],
                    value="auto",
                    id="encoder",
                )
                yield Button("Start", id="start", variant="success")
                yield Button("Cancel", id="cancel", variant="error", disabled=True)

        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Label("Logs")
                yield RichLog(id="log", highlight=False, markup=False)
                with Horizontal(id="status_row"):
                    yield LoadingIndicator(id="spinner")
                    yield Label("Idle", id="stage")
                yield ProgressBar(total=100, id="progress")

            with Vertical(id="right"):
                yield Label("Text preview (latest rendered frame)")
                with VerticalScroll(id="preview_box"):
                    yield Static("", id="preview", markup=False)

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#spinner", LoadingIndicator).display = False

    def append_log(self, text: str) -> None:
        self.query_one("#log", RichLog).write(text)

    def set_stage(self, stage: str, spinning: bool) -> None:
        self._stage_text = stage
        self.query_one("#stage", Label).update(stage)
        spinner = self.query_one("#spinner", LoadingIndicator)
        spinner.display = spinning

    def set_progress(self, pct: Optional[float]) -> None:
        bar = self.query_one("#progress", ProgressBar)
        if pct is None:
            bar.update(total=100, progress=0)
        else:
            bar.update(total=100, progress=max(0, min(100, int(pct))))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "start":
            self.start_job()
        elif bid == "cancel":
            self.cancel_job()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Users expect Enter to "go". Start when submitting the main fields.
        if event.input.id in {"source", "out"}:
            self.start_job()

    def action_start(self) -> None:
        self.start_job()

    def action_cancel(self) -> None:
        self.cancel_job()

    def start_job(self) -> None:
        source = self.query_one("#source", Input).value.strip()
        if not source:
            self.notify("Enter a YouTube URL or a local file path.")
            return

        try:
            fps = float(self.query_one("#fps", Input).value)
            width = int(float(self.query_one("#width", Input).value))
            braille_thresh = int(float(self.query_one("#braille_thresh", Input).value))
        except ValueError:
            self.notify("FPS, Width, and Braille Thresh must be numbers.")
            return

        if fps <= 0:
            self.notify("FPS must be > 0.")
            return
        if width <= 0:
            self.notify("Width must be > 0.")
            return
        if braille_thresh < 0 or braille_thresh > 255:
            self.notify("Braille Thresh must be between 0 and 255.")
            return

        mode = self.query_one("#mode", Select).value or "braille"
        out_name = self.query_one("#out", Input).value.strip() or "ascii_final.mp4"
        encoder = self.query_one("#encoder", Select).value or "auto"

        self.query_one("#start", Button).disabled = True
        self.query_one("#cancel", Button).disabled = False
        self.query_one("#log", RichLog).clear()
        self.query_one("#preview", Static).update("")
        self.set_stage("Starting...", True)
        self.set_progress(0)

        self.run_pipeline(source, fps, width, braille_thresh, mode, out_name, encoder)

    def cancel_job(self) -> None:
        for worker in list(self.workers):
            worker.cancel()
        self.append_log("Cancel requested.")
        self.set_stage("Cancelling...", True)


    @work(thread=True, exclusive=True, exit_on_error=False)
    def run_pipeline(
        self,
        source: str,
        fps: float,
        width: int,
        braille_thresh: int,
        mode: str,
        out_name: str,
        encoder: str,
    ) -> None:
        def post(msg: Message) -> None:
            self.post_message(msg)

        try:
            ensure_ffmpeg()

            workdir = Path("ascii_work")
            frames_dir = workdir / "frames"
            render_dir = workdir / "render"
            safe_mkdir(workdir)

            font_path = find_font_for_mode("braille" if mode == "braille" else "ascii")
            font_size = 14
            padding = 12

            # Clear old
            if frames_dir.exists():
                rmtree_retry(frames_dir)
            if render_dir.exists():
                rmtree_retry(render_dir)

            # 1) Get video
            if re.match(r"^https?://", source, re.I):
                post(LogLine("Downloading source..."))
                video_path = ytdlp_download(source, workdir, post, quiet=True)
            else:
                video_path = Path(source).expanduser().resolve()
                if not video_path.exists():
                    post(LogLine(f"File not found: {video_path}"))
                    return
                post(LogLine(f"Using local file: {video_path}"))

            # 2) Extract frames
            extract_frames(video_path, frames_dir, fps, post)

            # 3) Render frames
            render_frames(
                frames_dir=frames_dir,
                render_dir=render_dir,
                width_chars=width,
                braille_threshold=braille_thresh,
                mode="braille" if mode == "braille" else "ascii",
                font_path=font_path,
                font_size=font_size,
                padding=padding,
                post=post,
            )

            # 4) Encode and mux
            out_silent = workdir / "ascii_silent.mp4"
            out_final = Path(out_name).expanduser().resolve()
            assemble_video(render_dir, fps, out_silent, post, encoder=encoder)
            mux_audio(out_silent, video_path, out_final, post)

        except PipelineCancelled:
            post(LogLine("Cancelled."))
        except Exception as e:
            post(LogLine(f"Error: {e}"))

    # ---- UI thread handlers ----

    def on_log_line(self, msg: LogLine) -> None:
        self.append_log(msg.text)

    def on_progress_update(self, msg: ProgressUpdate) -> None:
        stage_map = {
            "download": ("Downloading...", True),
            "extract": ("Extracting frames...", True),
            "render": ("Rendering...", True),
            "encode": ("Encoding video...", True),
            "mux": ("Muxing audio...", True),
        }
        stage_text, spinning = stage_map.get(msg.stage, (msg.stage, True))
        self.set_stage(stage_text, spinning)
        self.set_progress(msg.percent)

        if msg.stage == "mux" and msg.percent == 100.0:
            self.set_stage("Done", False)
            self.query_one("#start", Button).disabled = False
            self.query_one("#cancel", Button).disabled = True

    def on_preview_frame(self, msg: PreviewFrame) -> None:
        self.query_one("#preview", Static).update(f"[Frame {msg.frame}/{msg.total}]\n\n{msg.text}")

    def on_worker_state_changed(self, event) -> None:
        # Textual's event type changed across versions; keep it tolerant.
        worker = getattr(event, "worker", None)
        if worker is not None and getattr(worker, "is_finished", False):
            self.query_one("#start", Button).disabled = False
            self.query_one("#cancel", Button).disabled = True
            # If we already reached "Done" via mux completion, keep it visible.
            if self._stage_text != "Done":
                self.set_stage("Idle", False)
            else:
                # Ensure the spinner is off if we finished successfully.
                self.query_one("#spinner", LoadingIndicator).display = False


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--headless", action="store_true", help="Run the pipeline without the TUI (for testing).")
    ap.add_argument("--source", type=str, default="", help="YouTube URL or local file path.")
    ap.add_argument("--fps", type=float, default=5.0, help="Output FPS / sampling rate (e.g. 5, 10, 24).")
    ap.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Seconds per frame (legacy). If provided, FPS will be computed as 1/interval.",
    )
    ap.add_argument("--width", type=int, default=200, help="Output width in characters.")
    ap.add_argument("--mode", choices=["ascii", "braille"], default="braille", help="Text mode.")
    ap.add_argument(
        "--encoder",
        choices=["auto", "libx264", "h264_nvenc"],
        default="auto",
        help="Video encoder for the intermediate MP4 (GPU helps encode speed, not text rendering).",
    )
    ap.add_argument(
        "--braille-threshold",
        type=int,
        default=140,
        help="Braille threshold 0-255 (lower = darker, higher = lighter).",
    )
    ap.add_argument("--out", type=str, default="ascii_final.mp4", help="Output mp4 path.")
    args = ap.parse_args()

    if args.headless:
        if not args.source:
            raise SystemExit("--source is required in --headless mode")

        def post(msg: Message) -> None:
            if isinstance(msg, LogLine):
                print(msg.text, flush=True)
            elif isinstance(msg, ProgressUpdate):
                pct = "" if msg.percent is None else f"{msg.percent:6.1f}%"
                print(f"[{msg.stage}] {pct}".rstrip(), flush=True)
            elif isinstance(msg, PreviewFrame):
                print(f"[preview] frame {msg.frame}/{msg.total}", flush=True)

        ensure_ffmpeg()
        workdir = Path("ascii_work")
        frames_dir = workdir / "frames"
        render_dir = workdir / "render"
        safe_mkdir(workdir)

        # Clear old
        if frames_dir.exists():
            rmtree_retry(frames_dir)
        if render_dir.exists():
            rmtree_retry(render_dir)

        font_path = find_font_for_mode("braille" if args.mode == "braille" else "ascii")
        font_size = 14
        padding = 12

        fps = float(args.fps)
        braille_thresh = int(args.braille_threshold)
        if args.interval is not None:
            if args.interval <= 0:
                raise SystemExit("--interval must be > 0")
            fps = 1.0 / float(args.interval)
        if fps <= 0:
            raise SystemExit("--fps must be > 0")
        if braille_thresh < 0 or braille_thresh > 255:
            raise SystemExit("--braille-threshold must be between 0 and 255")

        source = args.source.strip()
        if re.match(r"^https?://", source, re.I):
            post(LogLine("Downloading source..."))
            video_path = ytdlp_download(source, workdir, post, quiet=False)
        else:
            video_path = Path(source).expanduser().resolve()
            if not video_path.exists():
                raise SystemExit(f"File not found: {video_path}")
            post(LogLine(f"Using local file: {video_path}"))

        extract_frames(video_path, frames_dir, fps, post)
        render_frames(
            frames_dir=frames_dir,
            render_dir=render_dir,
            width_chars=int(args.width),
            braille_threshold=braille_thresh,
            mode="braille" if args.mode == "braille" else "ascii",
            font_path=font_path,
            font_size=font_size,
            padding=padding,
            post=post,
        )
        out_silent = workdir / "ascii_silent.mp4"
        out_final = Path(args.out).expanduser().resolve()
        assemble_video(render_dir, fps, out_silent, post, encoder=args.encoder)
        mux_audio(out_silent, video_path, out_final, post)
    else:
        AsciiMV().run()
