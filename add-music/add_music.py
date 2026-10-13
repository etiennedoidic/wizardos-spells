"""add_music — download audio from a URL into the local music library.

Takes a YouTube (or any other yt-dlp-supported) URL, downloads the audio, and
files it as ``<library>/<Artist>/<Album>/<Title>.<fmt>`` (FLAC by default) with
metadata tags written into the file — the layout Navidrome scans directly.

- Artist, title, album and date are auto-detected from yt-dlp metadata and the
  video title; any of them can be overridden by params.
- Multi-track sources are split into individual tracks: embedded chapters, an
  explicit ``timestamps`` param, or ``M:SS Title`` lines recovered from the
  video description. A playlist URL adds each entry as a track.
- Compilation mode (``as_playlist`` / ``playlist_name``) files each track under
  its own parsed artist with album ``Singles`` and writes an Extended M3U to
  ``<library>/Playlists/<name>.m3u8``.
- The library path comes from app settings (``music.library_path``, default
  ``~/Shared_Music``).
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from mutagen.flac import FLAC


# Patterns stripped from titles (case-insensitive)
_JUNK_SUFFIX_RE = re.compile(
    r"\s*[\(\[](official\s*)?(music\s*)?(video|audio|lyric|lyrics|visualizer|hd|hq|4k)[\)\]]",
    re.IGNORECASE,
)
_JUNK_PIPE_RE = re.compile(
    r"\s*\|\s*(official\s*)?(music\s*)?(video|audio|lyric|lyrics|visualizer|hd|hq|4k).*$",
    re.IGNORECASE,
)
# Common "show" names that appear as uploader but aren't the artist
_SHOW_NAMES = {
    "from the basement", "kexp", "tiny desk", "npr music", "colors",
    "colors studios", "cercle", "boiler room", "sofar sounds",
    "audiotree live", "paste magazine", "mahogany",
}


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = name.strip(". ")
    return name or "Unknown"


def _clean_title(raw: str) -> str:
    cleaned = _JUNK_SUFFIX_RE.sub("", raw)
    cleaned = _JUNK_PIPE_RE.sub("", cleaned)
    return cleaned.strip()


def _parse_artist_from_title(raw_title: str) -> tuple:
    """Try to extract (artist, remainder) from a video title.

    Handles patterns like:
      "Artist - Song Title"
      "Artist - Song | Show Name"
      "Artist - Album | Show Name"
    """
    for sep in [" - ", " – ", " — "]:
        if sep in raw_title:
            artist_part, rest = raw_title.split(sep, 1)
            rest = re.split(r"\s*\|\s*", rest)[0].strip()
            return artist_part.strip(), rest
    return "", raw_title


def _parse_metadata(info: dict) -> Dict[str, str]:
    """Extract artist/title/album/date from yt-dlp JSON."""
    raw_title = info.get("title", "")
    yt_artist = info.get("artist") or ""
    yt_creator = info.get("creator") or ""
    yt_uploader = info.get("uploader") or ""
    yt_track = info.get("track") or ""
    album = info.get("album") or ""
    date = info.get("release_year") or info.get("upload_date", "")[:4] or ""
    track_number = str(info.get("track_number") or "")

    title_artist, title_remainder = _parse_artist_from_title(raw_title)

    if yt_artist and yt_artist.lower() not in _SHOW_NAMES:
        artist = yt_artist
    elif yt_creator:
        creators = [c.strip() for c in yt_creator.split(",")]
        non_show = [c for c in creators if c.lower() not in _SHOW_NAMES]
        artist = non_show[0] if non_show else (title_artist or yt_uploader)
    elif title_artist:
        artist = title_artist
    elif yt_uploader and yt_uploader.lower() not in _SHOW_NAMES:
        artist = yt_uploader
    else:
        artist = title_artist or yt_uploader or ""

    title = yt_track or ""
    if not title:
        if title_artist:
            title = _clean_title(title_remainder)
        else:
            title = _clean_title(raw_title)

    return {
        "artist": artist or "Unknown Artist",
        "title": title or "Unknown Title",
        "album": album,
        "date": str(date),
        "track_number": track_number,
    }


# Matches a single timestamp line. Tolerates a leading bullet ("- ", "* ",
# "• ", etc.) and/or a track number ("1.", "01)", "1 -"), then a time in
# M:SS, MM:SS, H:MM:SS, or HH:MM:SS form, then the track title. Bullets
# are common in YouTube descriptions ("- 00:00 Only shallow"); track
# numbers are common in chapter blocks the user pastes manually.
_TIMESTAMP_LINE_RE = re.compile(
    r"^\s*"
    r"(?:[-*•·–—]\s+)?"
    r"(?:\d+\s*[\.\)\-]\s+)?"
    r"((?:\d{1,2}:)?\d{1,2}:\d{2})"
    r"\s*[\-–—:]?\s+"
    r"(.+?)\s*$"
)

# The other common shape: title first, time last ("01 - Elegant People
# 00:00"). Scanned with finditer so a run-on paste — a whole tracklist on
# one line, which is what a chat box that sends on Enter produces — still
# splits into tracks.
_TRAILING_TIMESTAMP_RE = re.compile(
    r"(?:^|\s)"
    r"(?:[-*•·–—]\s+)?"
    r"(?:\d+\s*[\.\)\-]\s+)?"
    r"(\S[^\n]*?)\s+"
    r"((?:\d{1,2}:)?\d{1,2}:\d{2})"
    r"(?=\s|$)"
)

# A DR-meter log line ("DR12      -0.03 dB   -15.43 dB      8:54 02-Foxey
# Lady"): the dB columns sit between the track's DR value and its time, so
# neither the time-first nor the time-last form above sees the time.
_DR_LOG_TIMESTAMP_RE = re.compile(
    r"^\s*DR\d+\s+[\d\.\-]+\s+dB\s+[\d\.\-]+\s+dB\s+"
    r"((?:\d{1,2}:)?\d{1,2}:\d{2})\s+"
    r"(.+?)\s*$"
)


def _hms_to_seconds(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 2:
        m, s = int(parts[0]), int(parts[1])
        return float(m * 60 + s)
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return float(h * 3600 + m * 60 + s)


def _parse_timestamps_block(text: str) -> List[Dict[str, Any]]:
    """Parse a multi-line block of "TIMESTAMP TITLE" lines into chapters.

    Lines that don't look like timestamps (preamble, blank lines, etc.) are
    silently skipped, so the caller can pass the raw text the user typed.

    Returns a list of `{"start_time": float, "title": str}` dicts sorted by
    start_time. End times are NOT filled in — call `_fill_chapter_ends` once
    the source duration is known.
    """
    chapters: List[Dict[str, Any]] = []
    for raw in (text or "").splitlines():
        m = _TIMESTAMP_LINE_RE.match(raw) or _DR_LOG_TIMESTAMP_RE.match(raw)
        if not m:
            continue
        start = _hms_to_seconds(m.group(1))
        title = m.group(2).strip().strip("-–— \t")
        if not title:
            continue
        chapters.append({"start_time": start, "title": title})

    if not chapters:
        # Time-last form, tried only when time-first found nothing — a
        # time-first block must never be re-parsed with its times glued to
        # the previous line's title.
        for m in _TRAILING_TIMESTAMP_RE.finditer(text or ""):
            title = re.sub(r"^\d+\s*[\.\)\-]\s+", "", m.group(1)).strip("-–— \t")
            if not title:
                continue
            chapters.append(
                {"start_time": _hms_to_seconds(m.group(2)), "title": title}
            )

    chapters.sort(key=lambda c: c["start_time"])
    return chapters


def _fill_chapter_ends(chapters: List[Dict[str, Any]], duration: Optional[float]) -> List[Dict[str, Any]]:
    """Set each chapter's `end_time` to the next chapter's start (last → duration)."""
    for i, ch in enumerate(chapters):
        if i + 1 < len(chapters):
            ch["end_time"] = chapters[i + 1]["start_time"]
        else:
            ch["end_time"] = float(duration) if duration else ch["start_time"] + 600.0
    return chapters


def _parse_chapter_title(chapter_title: str, fallback_artist: str) -> Dict[str, str]:
    """Parse a chapter title into artist + track name.

    Chapter titles can be:
      "Song Name"
      "Artist - Song Name"
      "01. Song Name"
      "1 - Song Name"
    """
    cleaned = re.sub(r"^\d+[\.\)\-\s]+\s*", "", chapter_title).strip()

    for sep in [" - ", " – ", " — "]:
        if sep in cleaned:
            artist, title = cleaned.split(sep, 1)
            return {"artist": artist.strip(), "title": title.strip()}

    return {"artist": fallback_artist, "title": cleaned or chapter_title}


def _tag_flac(path: str, meta: Dict[str, str]):
    audio = FLAC(path)
    audio["ARTIST"] = meta.get("artist", "")
    audio["TITLE"] = meta.get("title", "")
    audio["ALBUM"] = meta.get("album", "")
    if meta.get("date"):
        audio["DATE"] = meta["date"]
    if meta.get("track_number"):
        audio["TRACKNUMBER"] = meta["track_number"]
    audio.save()


def _place_file(
    src: Path, artist: str, album: str, title: str,
    track_num: str, fmt: str, library: Path,
) -> Path:
    safe_artist = _sanitize_filename(artist)
    safe_album = _sanitize_filename(album)
    safe_title = _sanitize_filename(title)

    if track_num:
        filename = f"{track_num.zfill(2)} {safe_title}.{fmt}"
    else:
        filename = f"{safe_title}.{fmt}"

    dest_dir = library / safe_artist / safe_album
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.move(str(src), str(dest))
    return dest


def _write_m3u_playlist(
    name: str, tracks: List[Dict[str, Any]], library: Path,
) -> Path:
    """Write an Extended M3U playlist under ``<library>/Playlists/<name>.m3u8``.

    Each track dict needs ``path`` (Path), ``artist`` (str), ``title`` (str),
    and ``duration`` (float seconds, optional). Track paths are written as
    paths relative to the playlist file so the library stays portable.
    Navidrome auto-discovers playlists anywhere under the library root.
    """
    safe_name = _sanitize_filename(name)
    playlists_dir = library / "Playlists"
    playlists_dir.mkdir(parents=True, exist_ok=True)
    dest = playlists_dir / f"{safe_name}.m3u8"

    lines = ["#EXTM3U", f"#PLAYLIST:{name}"]
    for t in tracks:
        try:
            duration = int(round(float(t.get("duration") or -1)))
        except (TypeError, ValueError):
            duration = -1
        artist = t.get("artist", "")
        title = t.get("title", "")
        try:
            rel = os.path.relpath(str(t["path"]), str(playlists_dir))
        except ValueError:
            rel = str(t["path"])
        lines.append(f"#EXTINF:{duration},{artist} - {title}")
        lines.append(rel)

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


class AddMusicTool:
    def __init__(self, music_config: Optional[Dict[str, Any]] = None,
                 progress: Optional[Any] = None, **_: Any):
        cfg = music_config or {}
        lib = cfg.get("library_path", "~/Shared_Music")
        self._library_path = Path(os.path.expanduser(lib))
        self._default_format = cfg.get("default_format", "flac")
        self._progress = progress or (lambda _msg: None)

    def _fetch_metadata(self, url: str) -> dict:
        self._progress("fetching metadata…")
        # --dump-single-json returns one object for both videos and playlists
        # (playlists get an `entries` array with full per-entry metadata).
        result = subprocess.run(
            ["yt-dlp", "--dump-single-json", "--no-download", url],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return json.loads(result.stdout)

    def _download_full(self, url: str, tmpdir: str) -> Path:
        self._progress("downloading audio…")
        fmt = self._default_format
        tmp_out = os.path.join(tmpdir, "audio.%(ext)s")
        cmd = [
            "yt-dlp", "-x",
            "--audio-format", fmt,
            "--audio-quality", "0",
            "--no-playlist",
            "-o", tmp_out,
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        found = list(Path(tmpdir).glob(f"audio.{fmt}"))
        if not found:
            found = [f for f in Path(tmpdir).iterdir() if f.is_file()]
        if not found:
            raise RuntimeError("download produced no files")
        return found[0]

    def _split_chapters(
        self, src: Path, chapters: List[dict], tmpdir: str,
    ) -> List[Path]:
        """Split a downloaded audio file by chapter timestamps using ffmpeg."""
        fmt = self._default_format
        parts = []
        for i, ch in enumerate(chapters):
            out = Path(tmpdir) / f"chapter_{i:03d}.{fmt}"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(src),
                "-ss", str(ch["start_time"]),
                "-to", str(ch["end_time"]),
                "-c:a", "flac" if fmt == "flac" else "copy",
                str(out),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg split failed for chapter {i+1}: {r.stderr[:200]}")
            parts.append(out)
        return parts

    def _handle_single(self, info: dict, url: str, overrides: dict) -> str:
        meta = _parse_metadata(info)
        meta.update({k: v for k, v in overrides.items() if v})
        if not meta["album"]:
            meta["album"] = "Singles"

        fmt = self._default_format
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._download_full(url, tmpdir)
            if fmt == "flac" and src.suffix == ".flac":
                _tag_flac(str(src), meta)
            dest = _place_file(
                src, meta["artist"], meta["album"], meta["title"],
                meta["track_number"], fmt, self._library_path,
            )

        return (
            f"Added to library:\n"
            f"  Artist: {meta['artist']}\n"
            f"  Title: {meta['title']}\n"
            f"  Album: {meta['album']}\n"
            f"  File: {dest}"
        )

    def _handle_chapters(
        self, info: dict, url: str, overrides: dict,
        playlist_name: Optional[str] = None,
    ) -> str:
        """Split a chaptered video into per-track files.

        When ``playlist_name`` is set, each track is filed under its own
        parsed artist (album = ``Singles``) and an M3U playlist threading the
        tracks in chapter order is written to ``<library>/Playlists/``.
        Otherwise, the whole video becomes a single album under one artist.
        """
        chapters = info["chapters"]
        base_meta = _parse_metadata(info)
        base_meta.update({k: v for k, v in overrides.items() if v})

        is_playlist_mode = bool(playlist_name)

        album = base_meta["album"] or _clean_title(info.get("title", "Unknown Album"))
        if overrides.get("album"):
            album = overrides["album"]
        artist = base_meta["artist"]

        fmt = self._default_format
        results = []
        placed_tracks: List[Dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._download_full(url, tmpdir)
            parts = self._split_chapters(src, chapters, tmpdir)

            for i, (part, ch) in enumerate(zip(parts, chapters)):
                ch_parsed = _parse_chapter_title(ch["title"], artist)
                track_artist = overrides.get("artist") or ch_parsed["artist"]
                track_title = ch_parsed["title"]
                track_num = str(i + 1)

                if is_playlist_mode:
                    track_album = "Singles"
                    file_track_num = ""
                else:
                    track_album = album
                    file_track_num = track_num

                meta = {
                    "artist": track_artist,
                    "title": track_title,
                    "album": track_album,
                    "date": base_meta["date"],
                    "track_number": file_track_num,
                }

                if fmt == "flac" and part.suffix == ".flac":
                    _tag_flac(str(part), meta)

                dest = _place_file(
                    part, track_artist, track_album, track_title,
                    file_track_num, fmt, self._library_path,
                )
                placed_tracks.append({
                    "path": dest,
                    "artist": track_artist,
                    "title": track_title,
                    "duration": float(ch.get("end_time", 0) - ch.get("start_time", 0)),
                })
                results.append(
                    f"  {track_num.zfill(2)}. {track_artist} - {track_title} -> {dest.name}"
                )
                self._progress(f"placed {track_num}/{len(chapters)}: {track_title}")

        if is_playlist_mode:
            playlist_path = _write_m3u_playlist(
                playlist_name, placed_tracks, self._library_path,
            )
            return (
                f"Added {len(results)} tracks and created playlist '{playlist_name}':\n"
                f"  Playlist: {playlist_path}\n"
                + "\n".join(results)
            )

        return (
            f"Added {len(results)} tracks to library:\n"
            f"  Artist: {artist}\n"
            f"  Album: {album}\n"
            + "\n".join(results)
        )

    def _handle_playlist(self, info: dict, overrides: dict) -> str:
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            return "Error: playlist has no playable entries"

        # YouTube auto-album playlists often title themselves "Album - Artist".
        # Strip the trailing " - Artist" if it matches the dominant entry artist.
        playlist_title = _clean_title(info.get("title") or "")
        playlist_uploader = info.get("uploader") or ""

        entry_artists = [e.get("artist") or "" for e in entries if e.get("artist")]
        dominant_artist = ""
        if entry_artists:
            dominant_artist = max(set(entry_artists), key=entry_artists.count)

        album = overrides.get("album") or playlist_title
        if dominant_artist and album.lower().endswith(f" - {dominant_artist.lower()}"):
            album = album[: -(len(dominant_artist) + 3)].strip()

        fmt = self._default_format
        results = []
        skipped = []

        for i, entry in enumerate(entries, start=1):
            entry_url = (
                entry.get("webpage_url")
                or entry.get("url")
                or (f"https://www.youtube.com/watch?v={entry['id']}" if entry.get("id") else None)
            )
            if not entry_url:
                skipped.append(f"  - entry {i}: no URL")
                continue

            try:
                meta = _parse_metadata(entry)
                meta["album"] = album
                if overrides.get("artist"):
                    meta["artist"] = overrides["artist"]
                if not meta.get("track_number"):
                    meta["track_number"] = str(entry.get("playlist_index") or i)

                with tempfile.TemporaryDirectory() as tmpdir:
                    src = self._download_full(entry_url, tmpdir)
                    if fmt == "flac" and src.suffix == ".flac":
                        _tag_flac(str(src), meta)
                    dest = _place_file(
                        src, meta["artist"], album, meta["title"],
                        meta["track_number"], fmt, self._library_path,
                    )
                results.append(
                    f"  {meta['track_number'].zfill(2)}. {meta['artist']} - {meta['title']} -> {dest.name}"
                )
                self._progress(f"placed {i}/{len(entries)}: {meta['title']}")
            except Exception as e:
                skipped.append(f"  - entry {i} ({entry.get('title', '?')}): {e}")

        artist_label = overrides.get("artist") or dominant_artist or playlist_uploader or "Various Artists"
        summary = (
            f"Added {len(results)} tracks from playlist to library:\n"
            f"  Artist: {artist_label}\n"
            f"  Album: {album}\n"
            + "\n".join(results)
        )
        if skipped:
            summary += "\nSkipped:\n" + "\n".join(skipped)
        return summary

    def execute(self, **kwargs: Any) -> str:
        url = kwargs.get("url", "").strip()
        if not url:
            return "Error: no URL provided"

        # Stripped: a stray leading/trailing space survives into the FLAC
        # tag even though _sanitize_filename trims it out of the folder
        # name, which shows up in Navidrome as a second, near-duplicate
        # artist. A whitespace-only override falls back to auto-detection.
        overrides = {
            "artist": kwargs.get("artist", "").strip(),
            "album": kwargs.get("album", "").strip(),
            "title": kwargs.get("title", "").strip(),
        }

        timestamps_text = (kwargs.get("timestamps") or "").strip()
        as_playlist = bool(kwargs.get("as_playlist"))
        playlist_name = (kwargs.get("playlist_name") or "").strip()
        if playlist_name:
            as_playlist = True

        try:
            info = self._fetch_metadata(url)
        except subprocess.TimeoutExpired:
            return "Error: metadata fetch timed out"
        except Exception as e:
            return f"Error fetching metadata: {e}"

        if timestamps_text:
            user_chapters = _parse_timestamps_block(timestamps_text)
            if not user_chapters:
                return (
                    "Error: `timestamps` was provided but no valid lines were "
                    "parsed. Each line must look like `M:SS Title` or `H:MM:SS Title`."
                )
            if len(user_chapters) < 2:
                return (
                    "Error: need at least two timestamps to split a video; got "
                    f"{len(user_chapters)}. Use a single-track add (omit "
                    "`timestamps`) instead."
                )
            duration = info.get("duration")
            info["chapters"] = _fill_chapter_ends(user_chapters, duration)

        # No explicit timestamps and no embedded chapters: try to recover
        # them from the video description. Many full-album and compilation
        # uploads list "M:SS Track" (or "- M:SS Track") lines there even
        # when YouTube didn't promote them to real chapters. This is useful
        # in both album mode (split into per-track files under one artist)
        # and playlist mode (split per parsed artist + emit M3U).
        if not timestamps_text and not (info.get("chapters") or []):
            desc_chapters = _parse_timestamps_block(info.get("description") or "")
            if len(desc_chapters) >= 2:
                info["chapters"] = _fill_chapter_ends(
                    desc_chapters, info.get("duration"),
                )

        try:
            if not timestamps_text and (info.get("_type") == "playlist" or "entries" in info):
                return self._handle_playlist(info, overrides)
            chapters = info.get("chapters") or []
            if as_playlist:
                if len(chapters) < 2:
                    return (
                        "Error: playlist mode needs at least two chapters or "
                        "timestamps. Provide `timestamps`, or use a video that "
                        "has chapters (or a description with `M:SS Title` lines)."
                    )
                resolved_name = (
                    playlist_name
                    or _clean_title(info.get("title") or "")
                    or "Mix"
                )
                return self._handle_chapters(
                    info, url, overrides, playlist_name=resolved_name,
                )
            if len(chapters) > 1:
                return self._handle_chapters(info, url, overrides)
            return self._handle_single(info, url, overrides)
        except Exception as e:
            return f"Error: {e}"


# ------------------------------------------------------------ wand seam


from pydantic import BaseModel, Field

from wizardos.sdk import App

from apps.wand.core.spells import spell

app_sdk = App("wand")


class AddMusicParams(BaseModel):
    url: str = Field(min_length=8, max_length=2000, pattern=r"^https?://",
                     description="YouTube (or other supported) URL")
    artist: str = Field(default="", description="Override artist (auto-detected)")
    title: str = Field(default="", description="Override title (auto-detected)")
    album: str = Field(default="", description="Override album (default: Singles)")
    timestamps: str = Field(
        default="",
        description="Optional 'M:SS Title' lines to split into tracks",
    )
    as_playlist: bool = Field(
        default=False,
        description="Compilation mode: file tracks per-artist + write an M3U",
    )
    playlist_name: str = Field(default="", description="M3U name (implies compilation)")


@spell(
    "add_music",
    "Download audio from a URL into the music library — tagged, filed "
    "Artist/Album, chapters and playlists become tracks",
    AddMusicParams,
)
def add_music(p: AddMusicParams, progress) -> str:
    # The tool reads library_path/default_format off this dict with the same
    # defaults, so app settings pass straight through.
    tool = AddMusicTool(music_config=app_sdk.settings.get("music") or {},
                        progress=progress)
    result = tool.execute(**p.model_dump())
    app_sdk.event("wand.add_music", {"url": p.url, "ok": not result.startswith("Error")})
    if result.startswith("Error"):
        # A failed download must look failed in the job table, not "done".
        raise RuntimeError(result)
    return result
