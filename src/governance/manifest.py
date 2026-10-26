# -*- coding: utf-8 -*-
"""Run manifest: everything needed to reproduce and audit a run."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

PACKAGES = ["pandas", "numpy", "statsmodels", "lightgbm", "scikit-learn", "scipy", "joblib"]


def _git_from_files(repo: Path) -> Dict[str, Any]:
    """Commit and branch read straight from .git (for schedulers whose PATH has no git executable)."""
    g = Path(repo) / ".git"
    head = (g / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return {"commit": head, "branch": "(detached)", "source": ".git files"}
    ref = head[5:]
    p = g / ref
    commit = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    if not commit and (g / "packed-refs").exists():
        for line in (g / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                commit = line.split()[0]
    return {"commit": commit, "branch": ref.rsplit("/", 1)[-1], "source": ".git files (git not on PATH)"}


def git_info(repo: Path) -> Dict[str, Any]:
    def _git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "describe": _git("describe", "--tags", "--always", "--dirty"),
                "dirty": bool(_git("status", "--porcelain", "--untracked-files=no"))}
    except Exception as exc:
        try:
            return _git_from_files(repo)
        except Exception:
            return {"error": f"{type(exc).__name__}: {exc}"}


def file_fingerprint(path: Path) -> Dict[str, Any]:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = path.stat()
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}


def package_versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


def write_manifest(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"generated": datetime.now().isoformat(timespec="seconds"), "platform": sys.platform, **payload}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
