"""Small web app (Gradio): pick a book, choose voices, preview, make the audiobook.

Long renders run in a background thread inside this process, so closing the
browser tab does not stop them; reopen the page to see progress again.
"""
from __future__ import annotations

import copy
import threading
import time
import traceback
from pathlib import Path

from .cli import _book_paths, load_or_analyze, make_engine
from .engines import ENGINES

SAMPLE_TEXT = "The rain had stopped at last. “Are you coming?” she asked, and for a moment nobody answered."


class Job:
    def __init__(self):
        self.thread: threading.Thread | None = None
        self.lines: list[str] = []
        self.status: dict = {}
        self.result: list[str] = []
        self.error = ""
        self.stop_requested = False

    def log(self, msg: str) -> None:
        self.lines.append(str(msg))
        self.lines = self.lines[-400:]

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


JOB = Job()
STATE: dict = {"script": None, "cover": None, "paths": None, "book": None}


def _settings_for(base: dict, engine: str, director: str, mode: str, voice: str | None) -> dict:
    s = copy.deepcopy(base)
    s["general"].update(engine=engine, director=director, mode=mode)
    # uploads live in a temp folder, so never save "next to the book" from the web app
    s["general"]["output_dir"] = s["general"].get("output_dir") or str(Path.home() / "Audiobooks")
    if voice:
        s[engine]["narrator"] = voice
    return s


def launch(settings: dict, port: int = 7860, share: bool = False, open_browser: bool = True) -> None:
    import gradio as gr

    def voice_choices(engine_name: str):
        try:
            eng = make_engine(_settings_for(settings, engine_name, "heuristic", "cast", None), log=lambda *_: None)
            vs = sorted(eng.voices(), key=lambda v: (-v.quality, v.gender, v.id))
            return [(f"{v.id}  ({v.gender}{', ' + v.accent if v.accent else ''})", v.id) for v in vs], eng.default_narrator()
        except Exception as e:  # noqa: BLE001 - e.g. no API key yet
            return [(f"(cannot list voices: {e})", "")], ""

    # ----------------------------------------------------------- actions --
    def read_book(file, director, engine):
        if file is None:
            return "Choose a book first.", gr.update(choices=[], value=[]), []
        path = Path(file if isinstance(file, str) else file.name)
        s = _settings_for(settings, engine, director, "cast", None)
        logs: list[str] = []
        try:
            script, cover, paths = load_or_analyze(path, s, reanalyze=False, assume_yes=True, log=logs.append)
        except Exception as e:  # noqa: BLE001
            return f"Could not read the book: {e}", gr.update(choices=[], value=[]), []
        STATE.update(script=script, cover=cover, paths=paths, book=path)
        titles = [f"{i + 1}. {ch.title}  ({ch.word_count():,} words)" for i, ch in enumerate(script.chapters)]
        chosen = [t for t, ch in zip(titles, script.chapters) if ch.include]
        script.count_lines()
        try:                                   # show who gets which voice right away
            from .casting import cast
            cast(script, make_engine(s, log=lambda *_: None), "cast")
            script.save(paths["script"])
        except Exception:  # noqa: BLE001 - e.g. cloud engine without a key yet
            pass
        rows = [[c.name, c.gender, c.lines, c.voices.get(engine, ""), c.description]
                for c in sorted(script.characters.values(), key=lambda c: -c.lines)]
        hours = script.word_count() / 155 / 60
        summary = (f"**{script.title}**" + (f" by {script.author}" if script.author else "") +
                   f"\n\n{len(script.chapters)} chapters, {script.word_count():,} words, about {hours:.1f} h of audio."
                   f"\n\nEditable script: `{paths['script']}`\n\n" + "\n".join(f"- {l}" for l in logs[-8:]))
        return summary, gr.update(choices=titles, value=chosen), rows

    def selected_chapters(chosen):
        return sorted(int(c.split(".", 1)[0]) - 1 for c in chosen or [])

    def apply_cast(rows, engine):
        script = STATE["script"]
        if script is None or rows is None:
            return
        data = rows.values.tolist() if hasattr(rows, "values") else rows
        for r in data:
            if r and r[0] in script.characters and len(r) > 3 and str(r[3] or "").strip():
                script.characters[r[0]].voices[engine] = str(r[3]).strip()

    def try_voice(engine, voice, text):
        from .engines.base import Part, Utterance
        try:
            eng = make_engine(_settings_for(settings, engine, "heuristic", "cast", voice), log=lambda *_: None)
            a = eng.synthesize(Utterance(voice=voice or eng.default_narrator(), speaker="sample",
                                         parts=[Part(text or SAMPLE_TEXT)]))
            return (eng.sample_rate, a), ""
        except Exception as e:  # noqa: BLE001
            return None, f"Could not render: {e}"

    pending = {"key": None, "t": 0.0}

    def estimate_cost(kind, s, chapters) -> float:
        """What this click would spend (0 for free engines)."""
        script = STATE["script"]
        cost = 0.0
        if s["general"]["director"] == "claude" and script.director != "claude":
            from .analyze.claude_director import estimate_cost as claude_cost
            cost += claude_cost(script, s["claude"]["model"])[0]
        if kind == "book":
            eng = make_engine(s, log=lambda *_: None)
            idx = chapters or [i for i, ch in enumerate(script.chapters) if ch.include]
            units = [u for i in idx for u in script.chapters[i].units]
            chars = sum(len(u.text) for u in units)
            words = sum(len(u.text.split()) for u in units)
            cost += eng.estimate_cost(chars, words / 155 * 60) or 0.0
        return cost

    def start(kind, engine, voice, mode, director, chosen, rows, preview_seconds, mp3):
        if JOB.running:
            return "A job is already running - see the log below."
        if STATE["script"] is None:
            return "Read a book first (step 1)."
        apply_cast(rows, engine)
        s = _settings_for(settings, engine, director, mode, voice)
        if mp3:
            s["general"]["formats"] = ["m4b", "mp3"]
        chapters = selected_chapters(chosen)
        try:
            cost = estimate_cost(kind, s, chapters)
        except Exception as e:  # noqa: BLE001
            return f"Could not start: {e}"
        if cost >= 0.05:              # paid: ask for a second click
            key = (kind, engine, director, tuple(chapters))
            if pending["key"] != key or time.time() - pending["t"] > 120:
                pending.update(key=key, t=time.time())
                return (f"**This will cost about ${cost:.2f}** (estimate). "
                        "Click the same button again within 2 minutes to go ahead.")
            pending["key"] = None
        JOB.__init__()
        JOB.thread = threading.Thread(target=_run, args=(kind, s, chapters, float(preview_seconds)), daemon=True)
        JOB.thread.start()
        return "Started..."

    def _run(kind, s, chapters, preview_seconds):
        from .export import export_m4b, export_mp3, export_wav_preview, safe_filename
        from .render import Renderer, RenderOptions
        from .textutil import Pronunciations
        try:
            script, cover, paths = STATE["script"], STATE["cover"], STATE["paths"]
            if s["general"]["director"] == "claude" and script.director != "claude":
                script, cover, paths = load_or_analyze(STATE["book"], s, reanalyze=True, assume_yes=True, log=JOB.log)
                STATE.update(script=script, cover=cover, paths=paths)
            g = s["general"]
            engine = make_engine(s, log=JOB.log)
            opts = RenderOptions(mode=g["mode"], pauses=s["pauses"], level_db=s["audio"]["level_db"],
                                 target_lufs=s["audio"]["target_lufs"], chapters=chapters or None,
                                 drop_dialogue_tags=bool(g.get("drop_dialogue_tags")),
                                 qa=bool(s["qa"]["enabled"]), qa_threshold=float(s["qa"]["threshold"]),
                                 retakes=int(s["qa"]["retakes"]))
            r = Renderer(script, engine, paths["work"], opts, Pronunciations.load(g.get("pronunciations")),
                         log=JOB.log, progress_cb=lambda st: JOB.status.update(st))
            script.save(paths["script"])
            if kind == "preview":
                JOB.log(f"Rendering a {preview_seconds:.0f}-second preview...")
                a = r.preview(preview_seconds, chapters[0] if chapters else None)
                out = export_wav_preview(a, engine.sample_rate, paths["out"] / "preview.mp3")
                JOB.result = [str(out)]
                JOB.log("Preview ready.")
                return
            est = r.estimate()
            JOB.log(f"{est['chapters']} chapters, about {est['audio_seconds'] / 3600:.1f} h of audio"
                    + (f", estimated cost ${est['cost_usd']:.2f}" if est["cost_usd"] is not None else ""))
            done = r.render()
            title = script.title or STATE["book"].stem
            outs = []
            if "m4b" in g["formats"]:
                out = paths["out"] / f"{safe_filename(title)}.m4b"
                export_m4b(done, out, title, script.author, cover, s["audio"]["bitrate"], paths["work"])
                outs.append(str(out))
            if "mp3" in g["formats"]:
                outs += [str(p) for p in export_mp3(done, paths["out"] / "mp3", title, script.author, cover)]
            JOB.result = outs
            JOB.log(f"Finished! Saved to {paths['out']}")
        except Exception as e:  # noqa: BLE001
            JOB.error = str(e)
            JOB.log("ERROR: " + str(e))
            JOB.log(traceback.format_exc(limit=3))

    shown = {"result": ()}

    def poll():
        st = JOB.status
        if JOB.running:
            pct = 100 * st.get("fraction", 0)
            eta = st.get("eta")
            head = f"Working: {pct:.1f}%  {st.get('chapter', '')}" + (
                f"  - about {int(eta // 3600)}h {int(eta % 3600 // 60)}m left" if eta else "")
        elif JOB.error:
            head = "Stopped with an error (see log)."
        elif JOB.result:
            head = "Done."
        else:
            head = "Idle."
        # only touch the players when the result changes, otherwise playback restarts every tick
        key = tuple(JOB.result)
        if key == shown["result"]:
            return head, "\n".join(JOB.lines[-60:]), gr.skip(), gr.skip()
        shown["result"] = key
        audio = JOB.result[0] if JOB.result and JOB.result[0].endswith((".mp3", ".wav")) else None
        files = [f for f in JOB.result if Path(f).is_file()] or None
        return head, "\n".join(JOB.lines[-60:]), audio, files

    def open_folder():
        import os
        import subprocess
        import sys
        if STATE["paths"]:
            p = str(STATE["paths"]["out"])
            if sys.platform == "win32":
                os.startfile(p)  # noqa: S606
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", p])

    # -------------------------------------------------------------- layout --
    g = settings["general"]
    first_choices, first_default = voice_choices(g["engine"])
    with gr.Blocks(title="Inkvoice") as app:
        gr.Markdown("# Inkvoice\nTurn a PDF / EPUB / TXT book into an expressive audiobook.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Book")
                book = gr.File(label="Book file", file_types=[".pdf", ".epub", ".txt", ".md"], type="filepath")
                director = gr.Radio([("Simple (free, offline)", "heuristic"), ("Claude (smart, paid)", "claude")],
                                    value=g["director"], label="Who decides speakers & emotions")
                read_btn = gr.Button("Read book", variant="primary")
                summary = gr.Markdown()
                chapters = gr.CheckboxGroup(label="Chapters to include", choices=[])
            with gr.Column(scale=1):
                gr.Markdown("### 2. Voices")
                engine = gr.Dropdown([(i.label, k) for k, i in ENGINES.items()], value=g["engine"], label="Voice engine")
                engine_info = gr.Markdown(f"Emotion: {ENGINES[g['engine']].emotion}")
                voice = gr.Dropdown(first_choices, value=first_default, label="Narrator voice", allow_custom_value=True)
                mode = gr.Radio([("A voice for every character", "cast"), ("One narrator for everything", "single")],
                                value=g["mode"], label="Style")
                sample = gr.Textbox(SAMPLE_TEXT, label="Try a voice with this text", lines=2)
                try_btn = gr.Button("Play sample")
                sample_audio = gr.Audio(label="Sample", interactive=False)
                sample_err = gr.Markdown()
        gr.Markdown("### Characters\nChange the **voice** column to recast someone (voice ids as in the list above).")
        cast_table = gr.Dataframe(headers=["character", "gender", "lines", "voice", "description"],
                                  datatype=["str", "str", "number", "str", "str"], interactive=True, wrap=True)
        gr.Markdown("### 3. Make it")
        with gr.Row():
            preview_len = gr.Slider(20, 300, value=60, step=10, label="Preview length (seconds)")
            mp3 = gr.Checkbox(label="Also make one MP3 per chapter", value="mp3" in g["formats"])
        with gr.Row():
            preview_btn = gr.Button("Preview")
            make_btn = gr.Button("Make audiobook", variant="primary")
            folder_btn = gr.Button("Open output folder")
        started = gr.Markdown()
        status = gr.Markdown("Idle.")
        out_audio = gr.Audio(label="Preview", interactive=False)
        out_files = gr.File(label="Results", file_count="multiple")
        log_box = gr.Textbox(label="Log", lines=12, max_lines=30, autoscroll=True)

        def on_engine(e):
            ch, d = voice_choices(e)
            return gr.update(choices=ch, value=d), f"Emotion: {ENGINES[e].emotion}"

        engine.change(on_engine, engine, [voice, engine_info])
        read_btn.click(read_book, [book, director, engine], [summary, chapters, cast_table], api_name="read_book")
        try_btn.click(try_voice, [engine, voice, sample], [sample_audio, sample_err], api_name="try_voice")
        args = [engine, voice, mode, director, chapters, cast_table, preview_len, mp3]
        preview_btn.click(lambda *a: start("preview", *a), args, started, api_name="preview")
        make_btn.click(lambda *a: start("book", *a), args, started, api_name="make")
        folder_btn.click(open_folder, None, None)
        timer = gr.Timer(2.0)
        timer.tick(poll, None, [status, log_box, out_audio, out_files], api_name="status")

    app.launch(server_port=port, share=share, inbrowser=open_browser, theme=gr.themes.Soft())
