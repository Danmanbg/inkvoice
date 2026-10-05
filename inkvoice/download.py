"""Model downloads from Hugging Face.

Some connections get very slow single streams from the Hugging Face CDN, so
large files are fetched as parallel byte ranges and verified against their
SHA-256 before use. Files land in <models_dir>/<repo name>/.
"""
from __future__ import annotations

import fnmatch
import hashlib
import http.client
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_MODELS_DIR = Path(os.environ.get("INKVOICE_MODELS", Path.home() / ".cache" / "inkvoice" / "models"))


def fetch_repo(repo: str, patterns: list[str] | None = None, models_dir: str | Path | None = None,
               connections: int = 8, piece: int = 8 << 20, log=print) -> Path:
    """Download the files of `repo` that match `patterns` (default: everything).

    Already-complete files are skipped, so calling this again is cheap.
    """
    from huggingface_hub import HfApi, hf_hub_url

    folder = Path(models_dir or DEFAULT_MODELS_DIR) / repo.split("/")[-1]
    marker = folder / ".complete"
    wanted_key = ",".join(sorted(patterns or ["*"]))
    if marker.is_file() and marker.read_text(encoding="utf-8") == wanted_key:
        return folder

    info = HfApi().model_info(repo, files_metadata=True)
    files = [s for s in info.siblings
             if not s.rfilename.startswith(".")
             and (patterns is None or any(fnmatch.fnmatch(s.rfilename, p) for p in patterns))]
    todo = []
    for s in files:
        dest = folder / s.rfilename
        if dest.is_file() and dest.stat().st_size == (s.size or 0):
            continue
        todo.append(s)

    if todo:
        total = sum(s.size or 0 for s in todo)
        got = [0]
        lock = threading.Lock()

        def fetch(url: str, path: Path, start: int, end: int) -> None:
            for attempt in range(12):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=60) as r, open(path, "r+b") as f:
                        f.seek(start)
                        while chunk := r.read(1 << 16):
                            f.write(chunk)
                            start += len(chunk)
                            with lock:
                                got[0] += len(chunk)
                    if start > end:
                        return
                except (OSError, http.client.HTTPException):
                    if attempt == 11:
                        raise
                    time.sleep(2 + attempt)

        jobs = []
        for s in todo:
            part = folder / (s.rfilename + ".partial")
            part.parent.mkdir(parents=True, exist_ok=True)
            with open(part, "wb") as f:
                f.truncate(s.size or 0)
            url = hf_hub_url(repo, s.rfilename)
            if not s.size:
                jobs.append((url, part, 0, 0))
            else:
                jobs += [(url, part, a, min(a + piece, s.size) - 1) for a in range(0, s.size, piece)]

        log(f"Downloading {repo} ({total / 1e6:.0f} MB) - this happens only once...")
        t0 = last_log = time.time()
        with ThreadPoolExecutor(connections) as pool:
            futures = [pool.submit(fetch, *j) for j in jobs]
            last = 0
            while not all(f.done() for f in futures):
                time.sleep(1)
                if time.time() - last_log >= 10:
                    speed = (got[0] - last) / (time.time() - last_log)
                    last, last_log = got[0], time.time()
                    eta = (total - got[0]) / speed / 60 if speed > 0 else 0
                    log(f"  {repo}: {100 * got[0] / max(1, total):.0f}%  {speed / 1e6:.2f} MB/s  ~{eta:.0f} min left")
            for f in futures:
                f.result()

        for s in todo:
            part = folder / (s.rfilename + ".partial")
            if s.lfs:
                h = hashlib.sha256()
                with open(part, "rb") as f:
                    while block := f.read(1 << 22):
                        h.update(block)
                if h.hexdigest() != s.lfs.sha256:
                    part.unlink(missing_ok=True)
                    raise RuntimeError(f"{repo}/{s.rfilename} was corrupted while downloading - run again.")
            part.replace(folder / s.rfilename)
        log(f"Downloaded {repo} in {(time.time() - t0) / 60:.1f} min.")

    folder.mkdir(parents=True, exist_ok=True)
    marker.write_text(wanted_key, encoding="utf-8")
    return folder


if __name__ == "__main__":       # python -m inkvoice.download <repo> [pattern ...] [--dir D]
    import sys
    args = sys.argv[1:]
    d = None
    if "--dir" in args:
        i = args.index("--dir")
        d = args[i + 1]
        del args[i:i + 2]
    print(fetch_repo(args[0], args[1:] or None, models_dir=d))
