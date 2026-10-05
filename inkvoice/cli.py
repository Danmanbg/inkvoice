"""Command line: inkvoice convert book.pdf  (see inkvoice --help)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__

EPILOG = """examples:
  inkvoice convert book.pdf                       free offline audiobook (Kokoro)
  inkvoice convert book.pdf --preview 90          hear the first 90 seconds first
  inkvoice convert book.pdf --director claude --engine gemini
  inkvoice analyze book.pdf                       only build the editable script
  inkvoice voices --engine kokoro --samples       render a sample of every voice
"""


def _book_paths(book: Path, settings: dict) -> dict:
    out_dir = Path(settings["general"].get("output_dir") or book.parent) / f"{book.stem} (audiobook)"
    return {"out": out_dir, "script": out_dir / f"{book.stem}.script.json", "work": out_dir / ".work"}


def _parse_chapters(spec: str | None, n: int) -> list[int] | None:
    if not spec:
        return None
    from .extract.pdf import parse_pages
    return sorted(parse_pages(spec, n) or [])


def _ask(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin or not sys.stdin.isatty():
        print(question + " -> not confirmed (run with --yes to allow).")
        return False
    return input(question + " [y/N] ").strip().lower() in ("y", "yes")


def _apply_overrides(settings: dict, args) -> None:
    g = settings["general"]
    for key in ("engine", "director", "mode"):
        v = getattr(args, key, None)
        if v:
            g[key] = v
    if getattr(args, "voice", None):
        settings[g["engine"]]["narrator"] = args.voice
    if getattr(args, "mp3", False):
        g["formats"] = sorted(set(g["formats"]) | {"mp3"})
    if getattr(args, "qa", False):
        settings["qa"]["enabled"] = True
    if getattr(args, "pages", None):
        settings["text"]["pages"] = args.pages
    if getattr(args, "out", None):
        g["output_dir"] = args.out
    if getattr(args, "variant", None):
        settings["chatterbox"]["variant"] = args.variant


def load_or_analyze(book: Path, settings: dict, reanalyze: bool = False, assume_yes: bool = False, log=print):
    from .analyze import build_script
    from .extract import load_book
    from .model import Script

    paths = _book_paths(book, settings)
    paths["out"].mkdir(parents=True, exist_ok=True)
    director = settings["general"]["director"]
    cover = None
    if paths["script"].is_file() and not reanalyze:
        script = Script.load(paths["script"])
        if director == "claude" and script.director != "claude":
            log("Script exists but was made by the simple director - re-analyzing with Claude.")
        else:
            log(f"Using the existing script {paths['script'].name} (edit it to change voices/emotions; "
                f"--reanalyze starts over).")
            cover_file = paths["work"] / "cover.jpg"
            return script, (cover_file.read_bytes() if cover_file.is_file() else None), paths

    log(f"Reading {book.name}...")
    t0 = time.time()
    last = [0.0]

    def prog(i, n):
        if time.time() - last[0] > 5:
            last[0] = time.time()
            log(f"  page {i}/{n}")
    bk = load_book(book, pages=settings["text"].get("pages") or None,
                   keep_footnotes=bool(settings["text"].get("keep_footnotes")), progress=prog)
    for note in bk.notes:
        log("  note: " + note)
    log(f"  {len(bk.chapters)} chapters, {bk.word_count():,} words ({time.time() - t0:.0f}s)")
    if not bk.chapters:
        raise SystemExit("No text found in this file.")
    opts = {}
    if director == "claude":
        c = settings["claude"]
        opts = dict(model=c["model"], effort=c["effort"], api_key=c.get("api_key") or None,
                    parallel=c.get("parallel", 4),
                    confirm=lambda cost: _ask(f"Use Claude to direct this book for about ${cost:.2f}?", assume_yes))
    script = build_script(bk, director=director, log=log, **opts)
    old = Script.load(paths["script"]) if paths["script"].is_file() else None
    if old:            # keep voices chosen earlier
        script.narrator_voices = old.narrator_voices or script.narrator_voices
        for name, c in script.characters.items():
            if name in old.characters and not c.voices:
                c.voices = old.characters[name].voices
    script.save(paths["script"])
    if bk.cover:
        paths["work"].mkdir(parents=True, exist_ok=True)
        (paths["work"] / "cover.jpg").write_bytes(bk.cover)
    log(f"Script saved: {paths['script']}")
    return script, bk.cover, paths


def print_cast(script, log=print):
    script.count_lines()
    chars = sorted(script.characters.values(), key=lambda c: -c.lines)
    if not chars:
        log("No speaking characters found.")
        return
    log("Characters (lines):")
    for c in chars[:25]:
        voices = ", ".join(f"{k}={v}" for k, v in c.voices.items())
        log(f"  {c.name:24} {c.gender:7} {c.lines:4}  {voices}  {c.description[:60]}")
    if len(chars) > 25:
        log(f"  ... and {len(chars) - 25} more")


def make_engine(settings: dict, log=print):
    from .engines import get_engine
    g = settings["general"]
    name = g["engine"]
    return get_engine(name, settings.get(name, {}), models_dir=g.get("models_dir") or None, log=log,
                      voices_dir=g.get("voices_dir"))


def cmd_convert(args, settings) -> int:
    from .export import export_m4b, export_mp3, export_wav_preview, safe_filename
    from .render import Renderer, RenderOptions
    from .textutil import Pronunciations

    book = Path(args.book).resolve()
    if not book.is_file():
        print(f"File not found: {book}")
        return 1
    script, cover, paths = load_or_analyze(book, settings, args.reanalyze, args.yes)
    if args.analyze_only:
        print_cast(script)
        return 0
    engine = make_engine(settings)
    g = settings["general"]
    opts = RenderOptions(
        mode=g["mode"], pauses=settings["pauses"], level_db=settings["audio"]["level_db"],
        target_lufs=settings["audio"]["target_lufs"], drop_dialogue_tags=bool(g.get("drop_dialogue_tags")),
        chapters=_parse_chapters(args.chapters, len(script.chapters)),
        qa=bool(settings["qa"]["enabled"]), qa_threshold=float(settings["qa"]["threshold"]),
        retakes=int(settings["qa"]["retakes"]))
    r = Renderer(script, engine, paths["work"], opts, Pronunciations.load(g.get("pronunciations")))
    script.save(paths["script"])                 # remember the voices that were cast
    print_cast(script)
    print(f"Narrator: {script.narrator_voices.get(engine.name)}   engine: {engine.name}   mode: {opts.mode}")

    if args.preview:
        a = r.preview(float(args.preview), (opts.chapters or [None])[0])
        out = export_wav_preview(a, engine.sample_rate, paths["out"] / "preview.mp3")
        print(f"\nPreview ready: {out}")
        return 0

    est = r.estimate()
    hours = est["audio_seconds"] / 3600
    print(f"\n{est['chapters']} chapters, {est['words']:,} words, about {hours:.1f} hours of audio.")
    if est["cost_usd"] is not None:
        if not _ask(f"{engine.name} will cost roughly ${est['cost_usd']:.2f}. Continue?", args.yes):
            return 1
    t0 = time.time()
    chapters = r.render()
    title = script.title or book.stem
    outputs = []
    if "m4b" in g["formats"]:
        out = paths["out"] / f"{safe_filename(title)}.m4b"
        export_m4b(chapters, out, title, script.author, cover, settings["audio"]["bitrate"], paths["work"])
        outputs.append(out)
    if "mp3" in g["formats"]:
        outputs += export_mp3(chapters, paths["out"] / "mp3", title, script.author, cover)[:1]
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")
    for o in outputs:
        print(f"  -> {o}")
    if (paths["work"] / "qa_report.txt").is_file():
        print(f"  Some lines may be garbled - see {paths['work'] / 'qa_report.txt'}")
    return 0


def cmd_voices(args, settings) -> int:
    from .engines import ENGINES
    from .engines.base import Part, Utterance
    from .export import export_wav_preview

    engine = make_engine(settings)
    info = ENGINES[engine.name]
    print(f"{info.label}\nEmotion: {info.emotion}\n")
    voices = engine.voices()
    for v in sorted(voices, key=lambda v: (v.gender, -v.quality, v.id)):
        print(f"  {v.id:28} {v.gender:8} {'*' * v.quality:6} {v.accent:4} {v.description}")
    if args.samples:
        out = Path(args.out or "voice samples") / engine.name
        out.mkdir(parents=True, exist_ok=True)
        text = args.text or ("The rain had stopped at last. “Are you coming?” she asked, "
                             "and for a moment nobody answered.")
        todo = [v for v in voices if not args.only or v.id in args.only.split(",")]
        print(f"\nRendering {len(todo)} samples into {out} ...")
        for v in todo:
            a = engine.synthesize(Utterance(voice=v.id, speaker="sample", parts=[Part(text)]))
            p = export_wav_preview(a, engine.sample_rate, out / f"{v.id.replace(':', '_')}.mp3")
            print("  ", p.name)
    return 0


def cmd_doctor(args, settings) -> int:
    import importlib
    from .audio import find_ffmpeg
    ok = True
    print(f"Inkvoice {__version__}   Python {sys.version.split()[0]}")
    print(f"Settings file: {settings.get('_file', '(none - using defaults)')}")
    try:
        print(f"ffmpeg: {find_ffmpeg()}")
    except RuntimeError as e:
        ok = False
        print(f"ffmpeg: MISSING - {e}")
    for mod, what in (("pdfplumber", "PDF reading"), ("pypdfium2", "PDF covers/bookmarks"), ("soundfile", "audio files"),
                      ("kokoro", "Kokoro engine"), ("chatterbox", "Chatterbox engine"), ("requests", "cloud engines"),
                      ("anthropic", "Claude director"), ("faster_whisper", "quality check"), ("gradio", "web app")):
        try:
            importlib.import_module(mod)
            print(f"  ok       {mod:16} {what}")
        except Exception as e:  # noqa: BLE001
            print(f"  missing  {mod:16} {what}  ({type(e).__name__}: {str(e).splitlines()[0][:80]})")
    for sec, env in (("gemini", "GEMINI_API_KEY"), ("elevenlabs", "ELEVENLABS_API_KEY"), ("claude", "ANTHROPIC_API_KEY")):
        print(f"  {sec:10} key: {'set' if settings[sec].get('api_key') else 'not set'} ({env})")
    return 0 if ok else 1


def cmd_gui(args, settings) -> int:
    from .gui import launch
    launch(settings, port=args.port, share=False, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inkvoice", description="Turn books into expressive audiobooks.",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"inkvoice {__version__}")
    p.add_argument("--settings", help="path to settings.toml")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--engine", choices=["kokoro", "chatterbox", "gemini", "elevenlabs"])
        sp.add_argument("--voice", help="narrator voice for the chosen engine")
        sp.add_argument("--mode", choices=["cast", "single"], help="cast = voice per character, single = one narrator")
        sp.add_argument("--variant", choices=["standard", "turbo"], help="Chatterbox model variant")
        sp.add_argument("--out", help="output folder (default: next to the book)")

    c = sub.add_parser("convert", help="make an audiobook from a PDF/EPUB/TXT")
    c.add_argument("book")
    common(c)
    c.add_argument("--director", choices=["heuristic", "claude"], help="who decides speakers and emotions")
    c.add_argument("--chapters", help="only these chapters, e.g. 1-3,7")
    c.add_argument("--pages", help="only these PDF pages, e.g. 5-320")
    c.add_argument("--preview", type=float, metavar="SECONDS", help="render just the first SECONDS")
    c.add_argument("--mp3", action="store_true", help="also write one MP3 per chapter")
    c.add_argument("--qa", action="store_true", help="listen back with Whisper and retake garbled lines")
    c.add_argument("--reanalyze", action="store_true", help="ignore the saved script and analyze again")
    c.add_argument("--yes", "-y", action="store_true", help="don't ask before spending money")
    c.set_defaults(func=cmd_convert, analyze_only=False)

    a = sub.add_parser("analyze", help="only build/refresh the editable script")
    a.add_argument("book")
    a.add_argument("--director", choices=["heuristic", "claude"])
    a.add_argument("--pages")
    a.add_argument("--out")
    a.add_argument("--reanalyze", action="store_true")
    a.add_argument("--yes", "-y", action="store_true")
    a.set_defaults(func=cmd_convert, analyze_only=True, engine=None, voice=None, mode=None, chapters=None,
                   preview=None, mp3=False, qa=False, variant=None)

    v = sub.add_parser("voices", help="list voices (and optionally render samples)")
    common(v)
    v.add_argument("--samples", action="store_true", help="render a short sample per voice")
    v.add_argument("--only", help="comma-separated voice ids to sample")
    v.add_argument("--text", help="sample text")
    v.set_defaults(func=cmd_voices)

    d = sub.add_parser("doctor", help="check that everything is installed")
    d.set_defaults(func=cmd_doctor)

    gui = sub.add_parser("gui", help="open the web app")
    gui.add_argument("--port", type=int, default=7860)
    gui.add_argument("--no-browser", action="store_true")
    gui.set_defaults(func=cmd_gui)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and Path(argv[0]).suffix.lower() in (".pdf", ".epub", ".txt", ".md"):
        argv.insert(0, "convert")              # drag & drop: inkvoice book.pdf
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    from .config import load_settings
    settings = load_settings(args.settings)
    _apply_overrides(settings, args)
    try:
        return args.func(args, settings)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where you left off.")
        return 130
    except Exception as e:  # noqa: BLE001
        from .engines.base import EngineError
        if isinstance(e, (EngineError, RuntimeError)):
            print(f"\nError: {e}")
            return 2
        raise
