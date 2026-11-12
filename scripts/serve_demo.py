"""Start the API in demo mode (no database, no keys), even if .env says live.

Usage: uv run python scripts/serve_demo.py [--port 8000]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    os.environ["EVIDENCE_RAG_MODE"] = "demo"  # environment wins over .env in pydantic-settings
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=args.port,
        app_dir=str(Path(__file__).resolve().parent.parent),
    )
