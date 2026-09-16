"""Run provenance: the metadata that answers "why is the model in production
the way it is?" six months later, when whoever trained it has left.

Four things must be versioned together -- code, data, model, config. This
module captures identifiers for the first two so MLflow can record them
alongside the third and fourth.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_sha() -> str:
    """Commit the training code was run from, or 'uncommitted'."""
    return _git("rev-parse", "HEAD") or "uncommitted"


def git_dirty() -> bool:
    """True if tracked files differ from HEAD.

    A dirty run is not reproducible: the SHA you logged does not describe the
    code that actually ran. Worth a loud warning, not a hard failure -- you
    experiment dirty all day, you just must not *promote* a dirty run.
    """
    status = _git("status", "--porcelain")
    return bool(status)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash of a data file.

    This is the data version. Stage 03 replaces it with DVC, which does the
    same thing with a pointer file and remote storage -- the idea is already
    here, just without the tooling.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def collect(data_path: Path) -> dict[str, str]:
    """Everything worth stamping onto a run."""
    return {
        "git_sha": git_sha(),
        "git_dirty": str(git_dirty()).lower(),
        "data_file": data_path.name,
        "data_sha256": file_sha256(data_path),
    }
