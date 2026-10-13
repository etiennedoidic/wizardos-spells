"""Transcribe a YouTube video.

Read the caption track first. If YouTube has no track, or returns HTTP 429,
transcribe the audio with whisper-cli. Print the text and write JSON to the
cache directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Grimoire used to reject a note above this size. Keep the same cut for the text file.
_MAX_ENTRY_CHARS = 100_000

# Retry a short network fault. A missing caption track starts the audio fallback.
_ATTEMPTS = 3
_BACKOFF_S = 2.0
_TIMEOUT_S = 180

_AUDIO_TIMEOUT_S = 900

# Public ggml file, English only, 148 MB. The download needs no token.
_WHISPER_MODEL = "ggml-base.en.bin"
_WHISPER_LABEL = "base.en"
_WHISPER_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/" + _WHISPER_MODEL
)
_WHISPER_THREADS = 4
_WHISPER_MIN_TIMEOUT_S = 300
_WHISPER_MAX_TIMEOUT_S = 3600

_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)%")
_RATE_LIMITED_RE = re.compile(r"\b429\b|too\s+many\s+requests", re.I)
_NO_SUBS_RE = re.compile(r"no subtitles|subtitles.{0,20}not available", re.I)
# whisper.cpp writes these marks for silence. Drop them.
_NOISE_RE = re.compile(r"^[\[(][^\])]*[\])]$")
_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})?$")

_YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "www.youtu.be",
}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_RE = re.compile(r"^https?://([^/:?#]+)(?::\d+)?(/[^?#]*)?(?:\?([^#]*))?", re.I)
_AUDIO_EXT = {".m4a", ".mp4", ".webm", ".opus", ".ogg", ".mp3", ".wav", ".flac"}


class _NoCaptions(Exception):
    """No caption track, or HTTP 429. Start the audio fallback."""


def _cache_dir() -> Path:
    """Return the cache directory. ``YT_TRANSCRIBE_CACHE`` overrides it."""
    raw = os.environ.get("YT_TRANSCRIBE_CACHE")
    return Path(raw) if raw else Path.home() / ".cache" / "yt-transcribe"


def _video_id(url: str) -> str:
    """Return the 11-character video id, or raise ValueError."""
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


def _check(url: str, lang: str) -> None:
    """Reject a bad URL or language code before any network request."""
    if not (8 <= len(url) <= 2000) or not url.startswith(("http://", "https://")):
        raise ValueError("url must be an http(s) URL")
    if not _LANG_RE.match(lang):
        raise ValueError(f"bad language code: {lang}")
    _video_id(url)


def _cue(start_ms: float, end_ms: float, text: str) -> Dict[str, Any]:
    """Build one segment. Times are milliseconds."""
    start = start_ms / 1000.0
    return {
        "start": round(start, 3),
        "end": round(max(start, end_ms / 1000.0), 3),
        "text": text,
    }


def _ordered(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort cues. Stop each end at the next start."""
    cues.sort(key=lambda c: c["start"])
    for i, cue in enumerate(cues[:-1]):
        cue["end"] = round(max(cue["start"], min(cue["end"], cues[i + 1]["start"])), 3)
    return cues


def _parse_json3(raw: dict) -> List[Dict[str, Any]]:
    """Convert json3 events to segments. Drop window and aAppend events."""
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
    """Convert whisper.cpp JSON to the same segments."""
    cues: List[Dict[str, Any]] = []
    for item in raw.get("transcription") or []:
        text = " ".join((item.get("text") or "").split())
        if not text or _NOISE_RE.match(text):
            continue
        offsets = item.get("offsets") or {}
        cues.append(_cue(float(offsets.get("from", 0)), float(offsets.get("to", 0)), text))
    return _ordered(cues)


def _clock(seconds: float) -> str:
    """Format a time as M:SS, or H:MM:SS after one hour."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _run_yt_dlp(
    url: str, workdir: Path, extra: List[str], *,
    timeout_s: int, retry_label: str, gave_up: str,
    fatal: Callable[[str], None], progress, out_tmpl: str = "%(id)s",
) -> None:
    """Run yt-dlp in ``workdir``. Retry, then raise. ``fatal`` can stop the retries."""
    binary = shutil.which("yt-dlp")
    if binary is None:
        raise RuntimeError("yt-dlp is not installed — `brew install yt-dlp`")

    cmd = [
        binary, "--no-playlist", "--no-progress", "--write-info-json", *extra,
        "--retries", "3", "-o", str(workdir / out_tmpl), url,
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
    """Write captions into ``workdir``, or signal the audio fallback."""

    def fatal(last: str) -> None:
        # HTTP 429: do not retry. Start the audio fallback.
        if _RATE_LIMITED_RE.search(last):
            raise _NoCaptions("YouTube rate-limited the caption fetch (HTTP 429)")
        if _NO_SUBS_RE.search(last):
            raise _NoCaptions(f"no {lang} caption track published")

    _run_yt_dlp(
        url, workdir,
        [
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs", f"{lang},{lang}-orig,{lang}.*",
            "--sub-format", "json3",
        ],
        timeout_s=_TIMEOUT_S, retry_label="retry ",
        gave_up="could not fetch captions", fatal=fatal, progress=progress,
    )


def _pick_subtitles(workdir: Path, video_id: str, lang: str) -> Path:
    """Pick ``lang``, then ``lang-orig``, then a regional track."""
    files = sorted(workdir.glob(f"{video_id}*.json3"))
    if not files:
        raise _NoCaptions(f"no {lang} caption track published")
    return min(files, key=lambda f: (
        0 if f.name == f"{video_id}.{lang}.json3" else
        1 if f.name == f"{video_id}.{lang}-orig.json3" else
        2 if f.name.startswith(f"{video_id}.{lang}") else 3
    ))


def _track_language(path: Path, video_id: str) -> str:
    """Read the language code from ``<id>.<lang>.json3``."""
    stem = path.name[len(video_id) + 1:]
    return stem[: -len(".json3")] if stem.endswith(".json3") else stem


def _info(workdir: Path, video_id: str) -> dict:
    """Read the metadata sidecar. A missing file is an empty dict."""
    sidecar = workdir / f"{video_id}.info.json"
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _ffmpeg() -> Optional[str]:
    """Return ffmpeg when the binary runs. A broken install returns None."""
    binary = shutil.which("ffmpeg")
    if binary is None:
        return None
    try:
        done = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return binary if done.returncode == 0 else None


def _to_wav(audio: Path, workdir: Path) -> Path:
    """Make a 16 kHz mono wav. Use ffmpeg, or afconvert when ffmpeg does not run."""
    wav = workdir / f"{audio.stem}.16k.wav"
    ffmpeg = _ffmpeg()
    if ffmpeg is not None:
        done = subprocess.run(
            [ffmpeg, "-y", "-i", str(audio), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
            capture_output=True, text=True, timeout=600,
        )
        if done.returncode == 0 and wav.exists():
            return wav
    afconvert = shutil.which("afconvert")
    if afconvert is not None:
        done = subprocess.run(
            [afconvert, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(audio), str(wav)],
            capture_output=True, text=True, timeout=600,
        )
        if done.returncode == 0 and wav.exists():
            return wav
        detail = (done.stderr or done.stdout or "afconvert failed").strip()
        raise RuntimeError(detail[:300])
    raise RuntimeError("ffmpeg does not run, and afconvert is not available")


def _download_audio(url: str, workdir: Path, video_id: str, progress) -> Path:
    """Save the audio as a 16 kHz wav. The temp directory deletes it."""

    def fatal(last: str) -> None:
        if _RATE_LIMITED_RE.search(last):
            raise RuntimeError(
                "YouTube rate-limited the audio download too (HTTP 429) — "
                "wait a few minutes and run again"
            )

    _run_yt_dlp(
        url, workdir,
        ["-f", "ba[ext=m4a]/140/ba"],
        timeout_s=_AUDIO_TIMEOUT_S, retry_label="audio retry ",
        gave_up="could not download the audio", fatal=fatal, progress=progress,
        out_tmpl="%(id)s.%(ext)s",
    )

    files = sorted(
        p for p in workdir.iterdir()
        if p.is_file() and p.suffix.lower() in _AUDIO_EXT and video_id in p.name
    )
    if not files:
        raise RuntimeError("yt-dlp reported success but wrote no audio file")
    return _to_wav(files[0], workdir)


def _whisper_model(progress) -> Path:
    """Return the ggml file. Download it once into the cache. No token."""
    path = _cache_dir() / _WHISPER_MODEL
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    progress(f"downloading {_WHISPER_MODEL}")
    part = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(_WHISPER_URL, part)
        part.replace(path)
    except (OSError, urllib.error.URLError) as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {_WHISPER_MODEL}: {exc}") from exc
    return path


def _whisper(
    audio: Path, workdir: Path, duration_s: Optional[float], progress,
) -> List[Dict[str, Any]]:
    """Transcribe ``audio`` once. Write JSON with ``-oj`` and report progress."""
    binary = shutil.which("whisper-cli")
    if binary is None:
        raise RuntimeError(
            "whisper-cli is not installed — `brew install whisper-cpp`"
        )
    model = _whisper_model(progress)

    stem = workdir / f"{audio.stem}.whisper"
    timeout_s = min(
        _WHISPER_MAX_TIMEOUT_S, max(_WHISPER_MIN_TIMEOUT_S, int(duration_s or 0))
    )
    cmd = [
        binary, "-m", str(model), "-f", str(audio), "-l", "en",
        "-t", str(_WHISPER_THREADS), "-oj", "-of", str(stem),
        "-np",
        "-pp",
    ]

    # Kill the process group on timeout. A live child keeps the pipe open.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )

    killed: List[bool] = []

    def _stop() -> None:
        killed.append(True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
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
        except subprocess.TimeoutExpired:
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
    """Build the transcript document for one video."""
    upload = str(info.get("upload_date") or "")
    return {
        "video_id": video_id,
        "url": info.get("webpage_url") or url,
        "title": info.get("title") or video_id,
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration_s": info.get("duration"),
        "published": f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}" if len(upload) == 8 else "",
        "language": lang,
        "source": source,
        "model": _WHISPER_LABEL if source == "audio" else "",
        "automatic": source == "audio" or not (info.get("subtitles") or {}).get(lang),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segment_count": len(segments),
        "segments": segments,
        "text": " ".join(s["text"] for s in segments),
    }


def _note(doc: Dict[str, Any]) -> str:
    """Build the note: title, then text."""
    text = doc["text"]
    limit = _MAX_ENTRY_CHARS - len(doc["title"]) - 1
    if len(text) > limit:
        text = text[: max(0, limit)]
    return f"{doc['title']}\n{text}"


def _write_json(doc: Dict[str, Any], out: Optional[Path]) -> Optional[Path]:
    """Write the full JSON. A disk error does not fail the job."""
    try:
        path = out or (_cache_dir() / "transcripts" / f"{doc['video_id']}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except OSError:
        return None


def yt_transcribe(url: str, lang: str, progress, out: Optional[Path] = None) -> str:
    """Transcribe ``url``. Return the title and the spoken text."""
    _check(url, lang)
    video_id = _video_id(url)
    progress(f"fetching {lang} captions for {video_id}")

    with tempfile.TemporaryDirectory(prefix="yt_transcribe-") as tmp:
        workdir = Path(tmp)
        try:
            _fetch_captions(url, lang, workdir, progress)
            subtitles = _pick_subtitles(workdir, video_id, lang)
        except _NoCaptions as gap:
            source, lang = "audio", "en"
            progress(f"{gap} — transcribing the audio with whisper {_WHISPER_LABEL}")
            audio = _download_audio(url, workdir, video_id, progress)
            info = _info(workdir, video_id)
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
        url=url, video_id=video_id, info=info, lang=lang,
        segments=segments, source=source,
    )
    spoken = _clock(segments[-1]["end"])
    progress(f"{len(segments)} cues, {len(doc['text']):,} characters over {spoken}")

    saved = _write_json(doc, out)
    if saved is not None:
        progress(f"wrote {saved}")

    return _note(doc)


def main(argv: Optional[List[str]] = None) -> int:
    """Read the CLI args, transcribe, and print the text."""
    parser = argparse.ArgumentParser(description="Transcribe a YouTube video.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--lang", default="en", help="Caption language (default: en)")
    parser.add_argument("--out", type=Path, help="JSON output path")
    args = parser.parse_args(argv)

    def progress(msg: str) -> None:
        print(msg, file=sys.stderr)

    try:
        text = yt_transcribe(args.url, args.lang, progress, args.out)
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
