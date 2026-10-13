"""yt_transcribe — a YouTube video's transcript, as JSON, filed in Grimoire.

URL in, transcript out: yt-dlp pulls the caption track, it becomes timestamped
segments, and a readable note lands in Grimoire's ``yt_transcripts`` chapter.

Captions first, always. YouTube publishes a timed track for essentially every
video (manual where an uploader wrote one, ASR otherwise), and reading it is a
metadata fetch — seconds, no model, no audio on disk. Charter § 3: never use
tier N when N−1 suffices.

The fallback covers the two ways tier 0 comes back empty — no caption track,
or a rate-limited fetch (HTTP 429). Then yt-dlp pulls the audio and
whisper.cpp's ``whisper-cli`` transcribes it: tier 1, entered only after tier
0 actually failed, announced over ``progress`` because it turns a three-second
job into a minutes-long one. ``source`` records which tier won.

Whisper here is a subprocess that exits, not a resident model: ``base.en`` is
148 MB and English-only — the fallback's whole scope — so it never contends
for the 16 GB budget the way an Oracle-managed LLM would, and Charter § 2's
one-resident-model rule is not in play. No ffmpeg anywhere either; yt-dlp
extracts the audio itself and whisper-cli decodes mp3 natively.

Nothing here talks to Anthropic, so ``CLAUDE_CODE_OAUTH_TOKEN`` is never read
and a stale subscription token cannot 401 a transcribe — the failure Ouroboros
hit twice building this spell is designed out rather than retried. The one
credential in play is the platform's own ``WIZARD_API_TOKEN``, which the SDK
loopback client carries to Grimoire — reached over HTTP, never by import
(Charter § 1), and only through ``POST /chapters`` and ``POST /capture``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from wizardos.sdk import App

from apps.wand.core.spells import spell

app_sdk = App("wand")


CHAPTER = "yt_transcripts"
CHAPTER_ICON = "🎬"
CHAPTER_DESCRIPTION = (
    "Transcripts of YouTube videos, captured by the wand yt_transcribe "
    "spell. Each entry is one video: title, channel, source URL and the "
    "spoken text in timestamped blocks."
)

# Grimoire's CaptureIn caps content at 100k characters, which a long enough
# video (roughly four hours of speech) overruns — so the note is trimmed to
# fit. The transcript is never lost: the JSON result and the archive copy hold
# it untrimmed.
_MAX_ENTRY_CHARS = 100_000

# Network work is retried, briefly. Anything past a few seconds of trying is
# a real outage, and that deserves to fail the job rather than hold a worker
# slot. "This video has no captions" is not retried at all — it is the
# fallback's cue.
_ATTEMPTS = 3
_BACKOFF_S = 2.0
_TIMEOUT_S = 180

# Audio is a real download, not a metadata fetch, so it gets its own budget.
_AUDIO_TIMEOUT_S = 900

# whisper.cpp, English-only, 148 MB. Small on purpose: the fallback exists to
# rescue a job, not to hold a GPU. `wizard-state/models/whisper/` is the
# platform's weights dir (SDK `models_dir`) — backed up nowhere, re-pullable.
_WHISPER_MODEL = "ggml-base.en.bin"
_WHISPER_LABEL = "base.en"
_WHISPER_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/" + _WHISPER_MODEL
)
# Four of ten cores. The box is headless but it is also serving the PWAs.
_WHISPER_THREADS = 4
# Wall clock allowed per cast, scaled by video length. base.en on this M4 runs
# far faster than realtime, so 1× duration is a hang detector, not a budget.
_WHISPER_MIN_TIMEOUT_S = 300
_WHISPER_MAX_TIMEOUT_S = 3600

_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)%")
# yt-dlp's rate-limit line, in either of the shapes it prints it.
_RATE_LIMITED_RE = re.compile(r"\b429\b|too\s+many\s+requests", re.I)
_NO_SUBS_RE = re.compile(r"no subtitles|subtitles.{0,20}not available", re.I)
# whisper.cpp labels non-speech spans instead of leaving them blank. They are
# not spoken text and would read as noise in the note.
_NOISE_RE = re.compile(r"^[\[(][^\])]*[\])]$")

# In scope: YouTube. Every other host is out.
_YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "www.youtu.be",
}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_RE = re.compile(r"^https?://([^/:?#]+)(?::\d+)?(/[^?#]*)?(?:\?([^#]*))?", re.I)


class _NoCaptions(Exception):
    """Tier 0 came back empty — hand off to the audio fallback.

    Deliberately narrow: only "no caption track published" and "rate-limited"
    raise it. A private video, a dead link or a network outage stays a hard
    failure, because ten minutes of whisper would not fix any of them.
    """


def _video_id(url: str) -> str:
    """The 11-character video id, or raise with a message worth showing. Takes
    watch links, youtu.be, /shorts/, /embed/ — what a share sheet produces."""
    m = _URL_RE.match(url.strip())
    if not m:
        raise ValueError(f"not a URL: {url!r}")
    host, path, query = m.group(1).lower(), m.group(2) or "/", m.group(3) or ""
    if host not in _YOUTUBE_HOSTS:
        raise ValueError(
            f"{host} is not YouTube — yt_transcribe only handles YouTube videos"
        )

    if host.endswith("youtu.be"):
        candidate = path.strip("/").split("/")[0]
    else:
        pairs = (part.partition("=") for part in query.split("&"))
        candidate = next((value for key, _, value in pairs if key == "v"), "")
        if not candidate:
            segments = [s for s in path.split("/") if s]
            if len(segments) >= 2 and segments[0] in ("shorts", "embed", "live", "v"):
                candidate = segments[1]

    if not _ID_RE.match(candidate):
        raise ValueError(f"no YouTube video id in {url!r}")
    return candidate


def _cue(start_ms: float, end_ms: float, text: str) -> Dict[str, Any]:
    """One segment, from whichever source's millisecond clock. Both parsers
    land here so caption and audio transcripts are one shape downstream."""
    start = start_ms / 1000.0
    return {
        "start": round(start, 3),
        "end": round(max(start, end_ms / 1000.0), 3),
        "text": text,
    }


def _ordered(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort cues and clamp each end to the next start. ASR durations overlap
    the next cue on purpose (a word is still on screen while the next line
    appears); ordered and non-overlapping is what a seek needs."""
    cues.sort(key=lambda c: c["start"])
    for i, cue in enumerate(cues[:-1]):
        cue["end"] = round(max(cue["start"], min(cue["end"], cues[i + 1]["start"])), 3)
    return cues


def _parse_json3(raw: dict) -> List[Dict[str, Any]]:
    """json3 caption events → segments. Two shapes have to be dropped or the
    transcript doubles: the leading window-definition event (no ``segs``), and
    the ``aAppend`` events carrying a bare newline to scroll YouTube's rolling
    caption box."""
    cues: List[Dict[str, Any]] = []
    for event in raw.get("events") or []:
        if event.get("aAppend"):
            continue
        text = "".join(s.get("utf8", "") for s in (event.get("segs") or []))
        text = " ".join(text.split())
        if not text:
            continue
        start_ms = float(event.get("tStartMs", 0))
        cues.append(_cue(start_ms, start_ms + float(event.get("dDurationMs", 0)), text))
    return _ordered(cues)


def _parse_whisper(raw: dict) -> List[Dict[str, Any]]:
    """whisper.cpp `-oj` output → the same segments.

    ``offsets`` is milliseconds from the start of the audio, which is the start
    of the video — yt-dlp extracts the whole track, so whisper's clock is
    YouTube's clock and the timestamps are directly seekable. The
    ``timestamps`` strings alongside are the same numbers formatted.
    """
    cues: List[Dict[str, Any]] = []
    for item in raw.get("transcription") or []:
        text = " ".join((item.get("text") or "").split())
        if not text or _NOISE_RE.match(text):
            continue
        offsets = item.get("offsets") or {}
        cues.append(_cue(float(offsets.get("from", 0)), float(offsets.get("to", 0)), text))
    return _ordered(cues)


def _clock(seconds: float) -> str:
    """`M:SS` under an hour, `H:MM:SS` over it — how people read a timecode."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _run_yt_dlp(
    url: str, workdir: Path, extra: List[str], *,
    timeout_s: int, retry_label: str, gave_up: str,
    fatal: Callable[[str], None], progress,
) -> None:
    """Run yt-dlp into `workdir` with retries; raise if it never succeeds.

    Both fetches share this, including the salvage of yt-dlp's own ERROR line —
    the useful half of a noisy stderr. `fatal` reads that line and raises when
    a retry is pointless, the one place the two paths differ: a refused caption
    fetch has a fallback, a refused audio download does not.
    ``--write-info-json`` is on both, so metadata comes out of the same
    extraction and the audio path still has a title when the caption attempt
    died before writing a sidecar.
    """
    binary = shutil.which("yt-dlp")
    if binary is None:
        raise RuntimeError("yt-dlp is not installed — `brew install yt-dlp`")

    cmd = [
        binary, "--no-playlist", "--no-progress", "--write-info-json", *extra,
        "--retries", "3", "-o", str(workdir / "%(id)s"), url,
    ]

    last = ""
    for attempt in range(1, _ATTEMPTS + 1):
        if attempt > 1:
            progress(f"{retry_label}{attempt}/{_ATTEMPTS} after: {last[:120]}")
            time.sleep(_BACKOFF_S * (attempt - 1))
        try:
            done = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            last = f"yt-dlp timed out after {timeout_s}s"
            continue
        if done.returncode == 0:
            return
        stderr = done.stderr or ""
        errors = [ln for ln in stderr.splitlines() if "ERROR" in ln]
        last = (errors[-1] if errors else stderr.strip()) or "yt-dlp failed"
        fatal(last)

    raise RuntimeError(f"{gave_up}: {last}")


def _fetch_captions(url: str, lang: str, workdir: Path, progress) -> None:
    """Captions + metadata into `workdir`, or the fallback's cue."""

    def fatal(last: str) -> None:
        # A 429 is not worth three tries — retrying is what earned it. Go to
        # the audio fallback immediately: faster, and the one request YouTube
        # has not already refused.
        if _RATE_LIMITED_RE.search(last):
            raise _NoCaptions("YouTube rate-limited the caption fetch (HTTP 429)")
        if _NO_SUBS_RE.search(last):
            raise _NoCaptions(f"no {lang} caption track published")

    _run_yt_dlp(
        url, workdir,
        [
            "--skip-download",       # no audio or video ever touches the disk
            "--write-subs",          # a human-written track wins when there is one
            "--write-auto-subs",     # ASR otherwise
            "--sub-langs", f"{lang},{lang}-orig,{lang}.*",
            "--sub-format", "json3",
        ],
        timeout_s=_TIMEOUT_S, retry_label="retry ",
        gave_up="could not fetch captions", fatal=fatal, progress=progress,
    )


def _pick_subtitles(workdir: Path, video_id: str, lang: str) -> Path:
    """The best caption file yt-dlp wrote: exact language, then the
    original-language track, then any regional variant — ``en`` before
    ``en-orig`` before ``en-GB``.

    Nothing at all is the common no-captions shape: yt-dlp exits 0, warns,
    and simply writes no ``.json3``. That is a `_NoCaptions`, not an error.
    """
    files = sorted(workdir.glob(f"{video_id}*.json3"))
    if not files:
        raise _NoCaptions(f"no {lang} caption track published")
    return min(files, key=lambda f: (
        0 if f.name == f"{video_id}.{lang}.json3" else
        1 if f.name == f"{video_id}.{lang}-orig.json3" else
        2 if f.name.startswith(f"{video_id}.{lang}") else 3
    ))


def _track_language(path: Path, video_id: str) -> str:
    """`<id>.<lang>.json3` → `<lang>`."""
    stem = path.name[len(video_id) + 1:]
    return stem[: -len(".json3")] if stem.endswith(".json3") else stem


def _info(workdir: Path, video_id: str) -> dict:
    """yt-dlp's metadata sidecar. Absent metadata is not a failed transcript."""
    sidecar = workdir / f"{video_id}.info.json"
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _download_audio(url: str, workdir: Path, video_id: str, progress) -> Path:
    """The video's audio as mp3 in `workdir`, for whisper to read. It lives in
    the caller's TemporaryDirectory and dies with it; only the transcript is
    kept."""

    def fatal(last: str) -> None:
        # Nothing left to fall back to: captions were refused and so was the
        # audio. Say which, so waiting is obviously the fix.
        if _RATE_LIMITED_RE.search(last):
            raise RuntimeError(
                "YouTube rate-limited the audio download too (HTTP 429) — "
                "wait a few minutes and cast again"
            )

    _run_yt_dlp(
        url, workdir,
        [
            # The audio stream alone, extracted by yt-dlp's own ffmpeg call.
            # Quality 5 is VBR ~130 kbps, and whisper resamples to 16 kHz mono
            # regardless, so anything better is bandwidth spent on nothing.
            "--extract-audio", "--audio-format", "mp3", "--audio-quality", "5",
        ],
        timeout_s=_AUDIO_TIMEOUT_S, retry_label="audio retry ",
        gave_up="could not download the audio", fatal=fatal, progress=progress,
    )

    files = sorted(workdir.glob(f"{video_id}*.mp3"))
    if not files:
        raise RuntimeError("yt-dlp reported success but wrote no audio file")
    return files[0]


def _whisper_model() -> Path:
    """The ggml weights, or how to get them. No auto-pull: a spell should not
    quietly spend 148 MB of someone's bandwidth mid-job."""
    path = app_sdk.models_dir / "whisper" / _WHISPER_MODEL
    if not path.exists():
        raise RuntimeError(
            f"whisper model {_WHISPER_MODEL} is not in {path.parent} — fetch it "
            f"once with:\n  mkdir -p {path.parent} && curl -L -o {path} {_WHISPER_URL}"
        )
    return path


def _whisper(
    audio: Path, workdir: Path, duration_s: Optional[float], progress,
) -> List[Dict[str, Any]]:
    """Transcribe `audio` with whisper.cpp. One run, no retry.

    ``-l en`` rather than auto-detect, to match the English-only model. Output
    goes to a JSON sidecar (``-oj``) rather than being scraped off stdout,
    because stdout is also where the backend banner and the progress ticks
    land. Those ticks are worth relaying: this is the step that takes minutes,
    and a silent job looks like a hung one.
    """
    binary = shutil.which("whisper-cli")
    if binary is None:
        raise RuntimeError(
            "whisper.cpp is not installed — `brew install whisper-cpp`. Until "
            "then yt_transcribe can only read videos that publish captions."
        )
    model = _whisper_model()

    stem = workdir / f"{audio.stem}.whisper"
    timeout_s = min(
        _WHISPER_MAX_TIMEOUT_S, max(_WHISPER_MIN_TIMEOUT_S, int(duration_s or 0))
    )
    cmd = [
        binary, "-m", str(model), "-f", str(audio), "-l", "en",
        "-t", str(_WHISPER_THREADS), "-oj", "-of", str(stem),
        "-np",   # results only — the model dump is not progress
        "-pp",   # …except the percentage, which is
    ]

    # `start_new_session` puts whisper in its own process group so the
    # watchdog can kill the group. Killing just the child is not enough: any
    # surviving grandchild still holds the write end of the pipe, the read
    # loop below never sees EOF, and a timeout turns into the hang it exists
    # to prevent. Reading stays on this thread — `progress` is a sqlite write
    # the worker expects from the handler's thread, not from a pump thread.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )

    killed: List[bool] = []

    def _stop() -> None:
        killed.append(True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # already gone
            pass

    watchdog = threading.Timer(timeout_s, _stop)
    watchdog.daemon = True
    watchdog.start()
    tail: deque = deque(maxlen=8)
    step = 0
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if not line:
                continue
            tail.append(line)
            found = _PROGRESS_RE.search(line)
            if found:
                percent = int(found.group(1))
                if percent >= step + 10:
                    step = percent - percent % 10
                    progress(f"whisper {step}%")
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # EOF but not reaped — stop it anyway
            _stop()
            code = proc.wait(timeout=30)
    finally:
        watchdog.cancel()
        if proc.stdout is not None:
            proc.stdout.close()

    if killed:
        raise RuntimeError(f"whisper timed out after {timeout_s}s")
    if code != 0:
        raise RuntimeError(f"whisper failed ({code}): {' / '.join(tail)[:300]}")

    out = Path(f"{stem}.json")
    try:
        raw = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RuntimeError(f"whisper wrote no readable transcript: {e}") from e
    return _parse_whisper(raw)


def _document(
    *, url: str, video_id: str, info: dict, lang: str,
    segments: List[Dict[str, Any]], source: str = "captions",
) -> Dict[str, Any]:
    upload = str(info.get("upload_date") or "")
    return {
        "video_id": video_id,
        "url": info.get("webpage_url") or url,
        "title": info.get("title") or video_id,
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration_s": info.get("duration"),
        "published": f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}" if len(upload) == 8 else "",
        "language": lang,
        # Which tier produced the text. Anything grading it later wants to
        # know: the two are not equally trustworthy, and only `captions` is
        # reproducible from a metadata fetch.
        "source": source,
        "model": _WHISPER_LABEL if source == "audio" else "",
        # An empty `subtitles` map means nobody wrote captions by hand and
        # what we read is YouTube's ASR — worth recording, because ASR text
        # is worth trusting less. A whisper transcript is automatic by
        # definition: no human wrote that either.
        "automatic": source == "audio" or not (info.get("subtitles") or {}).get(lang),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segment_count": len(segments),
        "segments": segments,
        "text": " ".join(s["text"] for s in segments),
    }


def _note(doc: Dict[str, Any]) -> str:
    """The Grimoire entry: the video's title, then the consolidated text.

    Nothing else — no channel, timestamps or source metadata. The JSON result
    (and the archive copy on disk) is where those live for anything wanting
    them.
    """
    text = doc["text"]
    limit = _MAX_ENTRY_CHARS - len(doc["title"]) - 1  # +1 for the title's newline
    if len(text) > limit:
        text = text[: max(0, limit)]
    return f"{doc['title']}\n{text}"


def _archive(doc: Dict[str, Any]) -> Optional[Path]:
    """Keep the untrimmed JSON on wand's own disk. Best effort — a transcript
    that reached Grimoire is not a failure because the archive copy didn't."""
    try:
        out = app_sdk.state_dir / "yt_transcripts"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{doc['video_id']}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except OSError:
        return None


async def _file_in_grimoire(note: str, client_id: str) -> dict:
    """Ensure the chapter, then capture the note into it.

    Capture deliberately refuses to mint a chapter (a typo would silently
    become one), so the chapter is created first; a 409 means a previous cast
    already made it and is exactly as good as a 200.
    """
    async with app_sdk.client(timeout=30.0) as client:
        made = await client.post("/api/grimoire/chapters", json={
            "name": CHAPTER,
            "description": CHAPTER_DESCRIPTION,
            "icon": CHAPTER_ICON,
        })
        if made.status_code not in (200, 409):
            raise RuntimeError(_http_error(f"could not create the {CHAPTER} chapter", made))

        captured = await client.post("/api/grimoire/capture", json={
            "content": note,
            "chapter": CHAPTER,
            "type": "log",
            "client_id": client_id,
        })
        if captured.status_code != 200:
            raise RuntimeError(_http_error("could not save the transcript", captured))
        return captured.json().get("entry") or {}


def _http_error(what: str, response) -> str:
    """A failure the user can act on — especially the 401, which is the
    platform's own loopback token and not a model credential. Naming the
    right secret is the whole point: guessing wrong sends someone off
    regenerating a Claude subscription token that had nothing to do with it.
    """
    if response.status_code == 401:
        return (
            f"{what}: the loopback API rejected wand's token (401). Reset it with "
            "`wizard secret set WIZARD_API_TOKEN` and restart the server. (This "
            "spell reads no model credential — a Claude token cannot cause this.)"
        )
    return f"{what}: grimoire returned {response.status_code} {response.text[:200]}"


class YtTranscribeParams(BaseModel):
    url: str = Field(
        min_length=8, max_length=2000, pattern=r"^https?://",
        description="YouTube video URL",
    )
    lang: str = Field(
        default="en", max_length=12, pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})?$",
        description="Caption language (default: en)",
    )

    @field_validator("url")
    @classmethod
    def _is_a_youtube_video(cls, raw: str) -> str:
        """The handler's own check, run early enough that a Vimeo link is a 422
        under the form field instead of a job that fails four seconds in."""
        _video_id(raw)
        return raw


@spell(
    "yt_transcribe",
    "Transcribe a YouTube video — its captions when it has them, whisper on "
    f"the audio when it doesn't. JSON out, filed in Grimoire's {CHAPTER} "
    "chapter with timestamps",
    YtTranscribeParams,
)
def yt_transcribe(p: YtTranscribeParams, progress) -> str:
    video_id = _video_id(p.url)  # rejects non-YouTube before any network work
    progress(f"fetching {p.lang} captions for {video_id}")

    with tempfile.TemporaryDirectory(prefix="yt_transcribe-") as tmp:
        workdir = Path(tmp)
        try:
            _fetch_captions(p.url, p.lang, workdir, progress)
            subtitles = _pick_subtitles(workdir, video_id, p.lang)
        except _NoCaptions as gap:
            # Tier 0 is genuinely out — not broken, out.
            source, lang = "audio", "en"
            progress(f"{gap} — transcribing the audio with whisper {_WHISPER_LABEL}")
            audio = _download_audio(p.url, workdir, video_id, progress)
            info = _info(workdir, video_id)  # this run wrote its own sidecar
            progress(f"transcribing {audio.stat().st_size / 1e6:.1f} MB of audio")
            segments = _whisper(audio, workdir, info.get("duration"), progress)
        else:
            source = "captions"
            info = _info(workdir, video_id)
            try:
                raw = json.loads(subtitles.read_text(encoding="utf-8"))
            except ValueError as e:
                raise RuntimeError(f"caption track was not readable json3: {e}") from e
            lang = _track_language(subtitles, video_id)
            segments = _parse_json3(raw)

    if not segments:
        held = "audio held no speech" if source == "audio" else "caption track held no spoken text"
        raise RuntimeError(f"the {held}")

    doc = _document(
        url=p.url, video_id=video_id, info=info, lang=lang,
        segments=segments, source=source,
    )
    spoken = _clock(segments[-1]["end"])
    progress(f"{len(segments)} cues, {len(doc['text']):,} characters over {spoken}")

    archived = _archive(doc)
    if archived is not None:
        doc["archive_path"] = str(archived)

    progress(f"filing in grimoire/{CHAPTER}")
    # The handler runs in the worker's thread pool, so there is no loop in
    # this thread for asyncio.run to conflict with.
    entry = asyncio.run(_file_in_grimoire(
        _note(doc), f"wand-yt:{video_id}:{int(time.time())}"
    ))
    doc["grimoire"] = {
        "chapter": entry.get("chapter") or CHAPTER,
        "entry_id": entry.get("id", ""),
    }
    if archived is not None:
        _archive(doc)  # re-write with the entry id recorded

    how = f"whisper {_WHISPER_LABEL}" if source == "audio" else f"{lang} captions"
    app_sdk.event("wand.yt_transcribe", {
        "video_id": video_id,
        "title": doc["title"],
        "language": lang,
        "source": source,
        "segments": len(segments),
        "saved": True,
        "text": f"Transcribed “{doc['title']}” from {how} — {len(segments)} cues over {spoken}",
    })
    return f"{doc['title']}\n{doc['text']}"
