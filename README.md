# Inkvoice

**Turn PDF, EPUB and TXT books into expressive audiobooks**, with a different voice for each
character, emotion on every line, chapter markers and the cover art, in a single `.m4b` file
that any audiobook app can play.

Most text-to-speech tools read a whole book in one flat voice. Inkvoice works more like an audiobook
production:

1. **Reads the book properly.** Running headers, page numbers, footnotes and words hyphenated
   across lines are cleaned up; paragraphs, scene breaks and chapters are rebuilt, and PDF
   bookmarks are used when present.
2. **Directs it.** Each line is split into narration and dialogue, and each line of dialogue gets
   a speaker, an emotion, an intensity, a delivery (whisper / shout / ...) and an optional acting
   note. A free rule-based director is built in. Optionally, **Claude** can read the whole book and
   direct it like a human director would.
3. **Casts it.** The narrator and every character get their own voice; you can change any of them.
4. **Performs it** with the speech engine you choose, then levels the audio, adds natural pauses
   and normalizes loudness to the audiobook standard (-18 LUFS, -3 dB peak).

Everything goes through an editable **script file**. You can fix a speaker, tone down an emotion or
correct a pronunciation, and on the next run only the lines you changed are re-recorded. Long books
can be stopped and resumed at any time.

## Which voice engine should I use?

| Engine | Cost | Emotion | Speed | Notes |
|---|---|---|---|---|
| **Kokoro** (local) | free | pace + volume, distinct voices | ~1x real time on a laptop CPU, much faster on a desktop | Very clean and clear, but it can't really act. Works offline. |
| **Chatterbox** (local) | free | real emotion control + voice cloning | very slow without an NVIDIA GPU | Good acting; clone any voice from a 10-second clip. Use with a GPU. |
| **Gemini 3.8 Flash TTS** (cloud) | about $0.80 per audio hour (≈$8 for a 10-hour book) | natural-language direction per line | fast | **Best value.** #4 on the Artificial Analysis TTS arena (Oct 2026). |
| **ElevenLabs Eleven v4** (cloud) | about $80 per million characters (≈$40 per novel) | inline audio tags `[whispers] [sad] [laughs]` | fast | **Best quality.** #1 on the arena (Oct 2026). |

Prices are as of October 2026 (Gemini's rates double on 1 Jan 2027). Inkvoice prints an estimate
and asks before it spends anything.

**Recommendations:**
- Free: start with **Kokoro**, using a voice per character.
- Real emotion for little money: use **Gemini**, ideally with the **Claude director**.
- The best possible result: use **ElevenLabs v4** with the Claude director.

### The Claude director (optional)

The built-in director recognizes `"…," said Mira`, `she whispered`, adverbs, punctuation and
turn-taking. Claude goes further: it works out who is speaking from context, and how each line
should sound given what just happened in the story. It only returns *labels* for numbered lines,
so the text that gets read is always exactly the book's text.

It needs an [Anthropic API key](https://console.anthropic.com/). A typical 100k-word novel costs
roughly $3–6 with the default `claude-opus-5-5` at low effort, or about half that with
`claude-sonnet-5-5`. You see an estimate first. The director also has refusal fallback enabled
(`fallbacks: "default"`).

## Install

You need **Python 3.11 or 3.12** (Kokoro does not support 3.13 yet) and **ffmpeg**:
- Windows: `winget install Gyan.FFmpeg`
- macOS: `brew install ffmpeg`
- Linux: `apt install ffmpeg`

```bash
git clone https://github.com/<you>/inkvoice
cd inkvoice
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[kokoro,gui]"            # free local voices + web app
pip install -e ".[claude]"                # optional: Claude director
pip install -e ".[chatterbox]"            # optional: Chatterbox (big; wants an NVIDIA GPU)
pip install -e ".[qa]"                    # optional: Whisper quality check
```

On Windows you can instead double-click `windows\install.bat`, then use `windows\Inkvoice.bat`
(the web app) or drag a book onto `windows\Convert book.bat`.

Copy `settings.example.toml` to `settings.toml` to change the defaults and add API keys.
Environment variables also work: `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `ANTHROPIC_API_KEY`.
Models are downloaded on first use into `~/.cache/inkvoice/models`: Kokoro is about 330 MB,
Chatterbox about 3–4 GB.

## Use

```bash
inkvoice gui                                         # web app at http://localhost:7860
inkvoice convert book.pdf --preview 90               # listen to the first 90 s first
inkvoice convert book.pdf                            # whole book -> "book (audiobook)/<title>.m4b"
inkvoice convert book.pdf --engine gemini --director claude
inkvoice convert book.epub --chapters 1-3 --mp3      # some chapters, plus one MP3 per chapter
inkvoice convert book.pdf --mode single              # one narrator for everything
inkvoice voices --engine kokoro --samples            # an MP3 sample of every voice
inkvoice doctor                                      # check the installation
```

The output folder `book (audiobook)/` contains the audiobook, the editable script, and a `.work`
cache. Run the same command again after a crash or Ctrl+C and it continues where it stopped.

### Editing the script

`book.script.json` has one line per piece of text:

```json
{"kind": "dialogue", "text": "There's no bell on those rocks,", "para": 13, "speaker": "Tobias", "emotion": "solemn", "intensity": 2, "delivery": "soft", "note": "barely above a whisper"}
```

You can change `speaker`, `emotion`, `intensity` (1–3), `delivery` (`normal`, `soft`, `whisper`,
`loud`, `shout`) and `note`, or set `"include": false` on a chapter to skip it. To choose voices,
edit the `characters` section, e.g. `"voices": {"kokoro": "bm_george", "gemini": "Algenib"}`.
Then run `convert` again.

Emotions: neutral, happy, excited, amused, playful, tender, calm, sad, angry, fearful, anxious,
surprised, disgusted, sarcastic, cold, hesitant, pleading, determined, proud, tired, solemn,
suspense.

### Pronunciations

Put names the engine mispronounces in `pronunciations.txt`:

```
Hermione = her-MY-oh-nee
Daenerys = /dɪˈnɛəɹɪs/
```

A respelling is used by every engine. IPA between slashes is used by Kokoro; the other engines
keep the original word.

### Your own voices (Chatterbox)

Put 10–15-second WAV clips of clear speech in `voices/` (for example `voices/m_old_captain.wav`)
and use the file name as a voice. `k:<kokoro voice>` (for example `k:bm_george`) clones a
Kokoro voice, which gives you about 15 different people for free. Only clone voices you have
permission to use.

### Quality check

`--qa` transcribes every generated line with Whisper and compares it with the text.
Expressive engines occasionally skip or repeat words; bad takes are re-recorded, and anything
still off is listed in `.work/qa_report.txt`.

## Tips

- **Preview first** (`--preview 60`) to pick voices before committing hours of rendering.
- Scanned PDFs (images of pages) contain no text. Run them through OCR first, e.g. `ocrmypdf`.
- `--pages 9-310` skips front and back matter that the automatic detection missed.
- In `cast` mode, `drop_dialogue_tags = true` removes bare "he said." tags, the way dramatized
  audiobooks do.

## Troubleshooting

- **Windows "An Application Control policy has blocked this file"**: this is Smart App Control
  blocking an unsigned DLL from a freshly released package. Installing a slightly older release
  usually works: `pip install "spacy==3.8.7"` (needed by Kokoro) and `pip install "av==18.1.0"`
  (needed by Whisper) and `pip install "scikit-learn==1.6.1"` (needed by Chatterbox).
- **`No module named 'pkg_resources'`** (Chatterbox): run `pip install "setuptools<81"`.
- **Slow or stuck model downloads**: Inkvoice downloads models in parallel pieces and checks
  their SHA-256. If a download is interrupted, run the command again.

## Legal

Only convert books you own or have the right to use, for your personal listening. Cloud engines
are subject to their providers' terms. Chatterbox adds an inaudible watermark to its output.

Inkvoice is released under the [MIT License](LICENSE). The models it downloads have their own
licenses: Kokoro (Apache-2.0), Chatterbox (MIT), Whisper (MIT). Python libraries used: pdfplumber
(MIT), pypdfium2 (Apache-2.0/BSD), soundfile (BSD), gradio (Apache-2.0), anthropic (MIT).
