"""
SynapseOS development entry point.

Usage:
    python run.py                        # defaults: 0.0.0.0:8000
    python run.py --port 9000
    python run.py --reload               # auto-reload on file changes

Production:
    uvicorn synapseos.api.app:app --host 0.0.0.0 --port 8000 --workers 1
"""
from __future__ import annotations

import argparse
import uvicorn

from synapseos.core.config import get_settings


def main() -> None:
    cfg = get_settings()

    parser = argparse.ArgumentParser(description="SynapseOS")
    parser.add_argument("--host", default=cfg.api_host)
    parser.add_argument("--port", type=int, default=cfg.api_port)
    parser.add_argument("--reload", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(
        "synapseos.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=cfg.api_log_level.lower(),
    )


if __name__ == "__main__":
    main()
