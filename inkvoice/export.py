"""Package rendered chapters as an M4B audiobook (chapters + cover) and/or MP3 files."""
from __future__ import annotations

import re
from pathlib import Path

from . import audio as A


def _meta_escape(s: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", s)


def safe_filename(s: str, limit: int = 80) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", s).strip(" .")
    s = re.sub(r"\s+", " ", s)
    return s[:limit].strip() or "audiobook"


def export_m4b(chapters: list[tuple[str, Path]], out: Path, title: str, author: str,
               cover: bytes | None = None, bitrate: str = "64k", workdir: Path | None = None) -> Path:
    workdir = Path(workdir or out.parent)
    workdir.mkdir(parents=True, exist_ok=True)
    lst = workdir / "concat.txt"
    lst.write_text("".join(f"file '{str(p.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
                           for _, p in chapters), encoding="utf-8")
    meta = [";FFMETADATA1", f"title={_meta_escape(title)}", f"album={_meta_escape(title)}",
            f"artist={_meta_escape(author or 'Unknown')}", f"album_artist={_meta_escape(author or 'Unknown')}",
            "genre=Audiobook", "comment=Made with Inkvoice"]
    t = 0.0
    for name, p in chapters:
        d = A.duration(p)
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(t * 1000)}", f"END={int((t + d) * 1000)}",
                 f"title={_meta_escape(name)}"]
        t += d
    meta_path = workdir / "chapters.txt"
    meta_path.write_text("\n".join(meta) + "\n", encoding="utf-8")

    args = ["-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(meta_path)]
    cover_path = None
    if cover:
        cover_path = workdir / "cover.jpg"
        cover_path.write_bytes(cover)
        args += ["-i", str(cover_path)]
    args += ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
    if cover_path:
        args += ["-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.m4b")
    args += ["-c:a", "aac", "-b:a", bitrate, "-ac", "1", "-ar", "44100", "-movflags", "+faststart",
             "-f", "mp4", str(tmp)]
    A.run_ffmpeg(args)
    tmp.replace(out)
    return out


def export_mp3(chapters: list[tuple[str, Path]], folder: Path, title: str, author: str,
               cover: bytes | None = None, quality: int = 5) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    cover_path = None
    if cover:
        cover_path = folder / "cover.jpg"
        cover_path.write_bytes(cover)
    outs = []
    n = len(chapters)
    for i, (name, p) in enumerate(chapters, 1):
        out = folder / f"{i:03d} - {safe_filename(name, 60)}.mp3"
        args = ["-i", str(p)]
        if cover_path:
            args += ["-i", str(cover_path), "-map", "0:a", "-map", "1:v", "-c:v", "copy",
                     "-metadata:s:v", "title=Album cover", "-metadata:s:v", "comment=Cover (front)"]
        args += ["-c:a", "libmp3lame", "-q:a", str(quality), "-ac", "1", "-id3v2_version", "3",
                 "-metadata", f"title={name}", "-metadata", f"album={title}", "-metadata", f"artist={author}",
                 "-metadata", f"track={i}/{n}", "-metadata", "genre=Audiobook", str(out)]
        A.run_ffmpeg(args)
        outs.append(out)
    return outs


def export_wav_preview(a, sr: int, out: Path) -> Path:
    """Write a preview as MP3 (falls back to WAV)."""
    wav = out.with_suffix(".wav")
    A.write_audio(wav, a, sr)
    try:
        A.run_ffmpeg(["-i", str(wav), "-c:a", "libmp3lame", "-q:a", "4", str(out.with_suffix(".mp3"))])
        wav.unlink(missing_ok=True)
        return out.with_suffix(".mp3")
    except Exception:  # noqa: BLE001
        return wav
