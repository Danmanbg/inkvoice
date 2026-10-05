"""Audio helpers: trimming, leveling, silence, file io, ffmpeg."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


def resample(a: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(a) == 0:
        return a.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_in, sr_out)
        return resample_poly(a, sr_out // g, sr_in // g).astype(np.float32)
    except ImportError:
        n = int(round(len(a) * sr_out / sr_in))
        return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(max(0, int(round(seconds * sr))), dtype=np.float32)


def _frame_db(a: np.ndarray, sr: int, win: float = 0.02) -> np.ndarray:
    n = max(1, int(sr * win))
    frames = len(a) // n
    if frames == 0:
        return np.array([-120.0])
    x = a[: frames * n].reshape(frames, n)
    rms = np.sqrt(np.mean(x * x, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def trim(a: np.ndarray, sr: int, threshold_db: float = -48.0, lead: float = 0.03, tail: float = 0.06) -> np.ndarray:
    """Remove silence at both ends, keeping a little air."""
    if len(a) == 0:
        return a
    db = _frame_db(a, sr)
    peak = db.max()
    thr = max(threshold_db, peak - 45)
    voiced = np.where(db > thr)[0]
    if len(voiced) == 0:
        return a[:0]
    n = int(sr * 0.02)
    start = max(0, voiced[0] * n - int(lead * sr))
    end = min(len(a), (voiced[-1] + 1) * n + int(tail * sr))
    return a[start:end]


def level(a: np.ndarray, sr: int, target_db: float = -20.0, offset_db: float = 0.0) -> np.ndarray:
    """Normalize speech loudness (RMS of voiced frames), then apply a delivery offset."""
    if len(a) == 0:
        return a
    db = _frame_db(a, sr)
    voiced = db[db > db.max() - 30]
    if len(voiced) == 0:
        return a
    current = 10 * np.log10(np.mean(10 ** (voiced / 10)))
    gain_db = float(np.clip(target_db + offset_db - current, -18, 18))
    out = a * (10 ** (gain_db / 20))
    return soft_limit(out)


def soft_limit(a: np.ndarray, ceiling: float = 0.89) -> np.ndarray:
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    if peak <= ceiling:
        return a.astype(np.float32)
    over = np.abs(a) > ceiling * 0.8
    out = a.copy()
    knee = ceiling * 0.8
    s = np.sign(a[over])
    x = np.abs(a[over]) - knee
    out[over] = s * (knee + (ceiling - knee) * np.tanh(x / (ceiling - knee)))
    return out.astype(np.float32)


def fade(a: np.ndarray, sr: int, ms: float = 6.0) -> np.ndarray:
    n = min(len(a) // 2, int(sr * ms / 1000))
    if n <= 1:
        return a
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    a = a.copy()
    a[:n] *= ramp
    a[-n:] *= ramp[::-1]
    return a


def write_audio(path: str | Path, a: np.ndarray, sr: int) -> None:
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    sf.write(str(tmp), np.clip(a, -1, 1), sr, subtype="PCM_16")
    tmp.replace(path)


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a, sr


def duration(path: str | Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


# ------------------------------------------------------------------ ffmpeg --
def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError("ffmpeg was not found. Install it (Windows: winget install Gyan.FFmpeg, "
                       "macOS: brew install ffmpeg, Linux: apt install ffmpeg) or pip install imageio-ffmpeg.")


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-y", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + p.stderr[-2000:])
    return p


TRUE_PEAK = -3.0     # headroom before lossy encoding (AAC/MP3 overshoot ~2 dB); also the ACX limit


def loudnorm(src: Path, dst: Path, target_lufs: float = -18.0, true_peak: float = TRUE_PEAK) -> None:
    """Two-pass EBU R128 normalization of a whole chapter (keeps dynamics, linear gain)."""
    p = run_ffmpeg(["-i", str(src), "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=15:print_format=json",
                    "-f", "null", "-"])
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    if not m:
        shutil.copyfile(src, dst)
        return
    st = json.loads(m.group(0))
    if float(st.get("input_i", -70)) < -60:        # (near) silent - nothing to do
        shutil.copyfile(src, dst)
        return
    af = (f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=15:measured_I={st['input_i']}:"
          f"measured_TP={st['input_tp']}:measured_LRA={st['input_lra']}:measured_thresh={st['input_thresh']}:"
          f"offset={st['target_offset']}:linear=true")
    import soundfile as sf
    sr = sf.info(str(src)).samplerate
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    run_ffmpeg(["-i", str(src), "-af", af, "-ar", str(sr), "-c:a", "pcm_s16le", str(tmp)])
    tmp.replace(dst)
