"""Shared helpers for the dataset download notebooks."""
from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

# Rough bounding box for Banten province (EPSG:4326). Swap in the official
# AOI (BIG/geoBoundaries) once precision matters for the actual analysis.
BANTEN_BBOX = {
    "west": 105.05,
    "south": -7.05,
    "east": 106.75,
    "north": -5.65,
}

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest: Path, chunk_size: int = 1 << 16, headers: dict | None = None) -> Path:
    """Download one file with a progress bar. Skips it if already present."""
    log = get_logger("download")
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Already have it, skipping: %s", dest)
        return dest

    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.rename(dest)
    log.info("Done: %s", dest)
    return dest


HDFS_LAKE = "hdfs://localhost:9000/lake"   # the HDFS from infra/hdfs


def lake_base() -> str:
    """Base of the Spark data: an HDFS URI, or "" for the local datasets/ folder.

    Uses HDFS if the NameNode answers on localhost:9000, else the local disk. LAKE_URI overrides this: a URI, or "local".
    """
    uri = os.environ.get("LAKE_URI", "").strip()
    if uri:
        return "" if uri.lower() == "local" else uri.rstrip("/")
    try:
        with socket.create_connection(("localhost", 9000), timeout=0.5):
            return HDFS_LAKE
    except OSError:
        return ""


def lake_path(*parts: str) -> str:
    """Full path of a file under datasets/, e.g. lake_path("processed", "features", "tabular_features.csv")."""
    base = lake_base()
    if base:
        return "/".join([base, *[p.strip("/") for p in parts]])
    return str(PROJECT_ROOT.joinpath("datasets", *parts))
