"""Video encoding and container helpers (ffmpeg / ffprobe)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class FFmpegMissing(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise FFmpegMissing(f"{tool} not found on PATH; install ffmpeg")
    return path


class VideoWriter:
    """Streams BGR frames into an H.264 MP4 (yuv420p, browser-playable)."""

    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int = 20):
        self.path = Path(path)
        cmd = [
            _require("ffmpeg"), "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-g", str(max(1, fps * 2)), "-threads", "1", "-map_metadata", "-1",
            "-movflags", "+faststart", str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.count += 1

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        self.proc.stderr.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed: {err.strip()}")


def stream_sha256(path: Path) -> str:
    """SHA-256 over the encoded video packets only (unchanged by container metadata edits)."""
    out = subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "hash", "-hash", "sha256", "-"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not out.upper().startswith("SHA256="):
        raise RuntimeError(f"unexpected hash output: {out!r}")
    return out.split("=", 1)[1].lower()


def embed_comment(src: Path, dst: Path, comment: str) -> None:
    subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-map", "0", "-c", "copy",
         "-metadata", f"comment={comment}", "-movflags", "+faststart", str(dst)],
        check=True, capture_output=True,
    )


def read_tags(path: Path) -> dict:
    res = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-show_entries", "format_tags:format=format_name", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return {}
    data = json.loads(res.stdout or "{}")
    tags = (data.get("format") or {}).get("tags") or {}
    return {k.lower(): v for k, v in tags.items()}


def probe_video(path: Path) -> dict | None:
    res = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,codec_name,nb_frames,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return None
    streams = json.loads(res.stdout or "{}").get("streams") or []
    return streams[0] if streams else None


def read_frames(path: Path, max_frames: int = 240, stride: int = 1) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    try:
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append(f)
            i += 1
    finally:
        cap.release()
    return frames
