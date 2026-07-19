import ipaddress
import os
import socket
import tempfile
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from omnishotcut.engine import load_model, single_video_inference
from omnishotcut.label_correspondence import (
    unique_inter_label_mapping,
    unique_intra_label_mapping,
)


CHECKPOINT_PATH = os.environ.get(
    "OMNISHOTCUT_CHECKPOINT", "checkpoints/OmniShotCut_ckpt.pth"
)
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_BYTES", str(2 * 1024**3)))
DOWNLOAD_TIMEOUT_SECONDS = float(os.environ.get("DOWNLOAD_TIMEOUT_SECONDS", "300"))
CONTEXT_FRAMES = int(os.environ.get("OMNISHOTCUT_CONTEXT_FRAMES", "20"))

if not Path(CHECKPOINT_PATH).is_file():
    raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

MODEL, MODEL_ARGS = load_model(CHECKPOINT_PATH)
INTRA_ID2NAME = {value: key for key, value in unique_intra_label_mapping.items()}
INTER_ID2NAME = {value: key for key, value in unique_inter_label_mapping.items()}

app = FastAPI(title="OmniShotCut EAS API", version="1.0.0")


def _validate_public_https_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise HTTPException(400, "video_url must be an HTTPS URL")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror as exc:
        raise HTTPException(400, "video_url hostname cannot be resolved") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise HTTPException(400, "video_url must resolve to a public address")
    return value


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    size = 0
    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                raise HTTPException(413, "video exceeds MAX_VIDEO_BYTES")
            output.write(chunk)


async def _download_video(url: str, destination: Path) -> None:
    _validate_public_https_url(url)
    timeout = httpx.Timeout(DOWNLOAD_TIMEOUT_SECONDS)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                size = 0
                with destination.open("wb") as output:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_VIDEO_BYTES:
                            raise HTTPException(413, "video exceeds MAX_VIDEO_BYTES")
                        output.write(chunk)
    except httpx.HTTPError as exc:
        raise HTTPException(400, f"unable to download video_url: {exc}") from exc


def _result(video_path: Path) -> dict:
    ranges, intra_labels, inter_labels, _, fps = single_video_inference(
        video_path=str(video_path),
        model=MODEL,
        model_args=MODEL_ARGS,
        overlap_window_length=CONTEXT_FRAMES,
    )
    return {
        "fps": float(fps),
        "shots": [
            {
                "start_frame": int(frame_range[0]),
                "end_frame": int(frame_range[1]),
                "start_time": round(int(frame_range[0]) / fps, 3) if fps else None,
                "end_time": round(int(frame_range[1]) / fps, 3) if fps else None,
                "intra_label": INTRA_ID2NAME.get(
                    int(intra_labels[index]), str(int(intra_labels[index]))
                ),
                "inter_label": INTER_ID2NAME.get(
                    int(inter_labels[index]), str(int(inter_labels[index]))
                ),
            }
            for index, frame_range in enumerate(ranges)
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(
    video: Annotated[UploadFile | None, File()] = None,
    video_url: Annotated[str | None, Form()] = None,
) -> dict:
    if (video is None) == (video_url is None):
        raise HTTPException(400, "provide exactly one of video or video_url")

    suffix = Path(video.filename or "video.mp4").suffix if video else ".mp4"
    with tempfile.TemporaryDirectory(prefix="omnishotcut_") as temp_dir:
        video_path = Path(temp_dir) / f"input{suffix or '.mp4'}"
        if video is not None:
            await _save_upload(video, video_path)
        else:
            await _download_video(video_url or "", video_path)
        return _result(video_path)
