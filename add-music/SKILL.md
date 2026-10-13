---
name: add-music
description: Download audio from a URL into a Navidrome-layout music library as tagged FLAC, splitting chapters and playlists.
---

# add-music

When someone wants a song, album or mix from a URL in their local music library.

A spell: one file, `add_music.py`, declared with `@spell("add_music", …)`. Wand renders its form from
the params model and runs it in the worker as a `wand.cast` job, streaming progress lines to the phone.

## Params

| Param | Meaning |
|---|---|
| `url` | YouTube (or other yt-dlp) URL, required |
| `artist`, `title`, `album` | Overrides; auto-detected otherwise (album defaults to Singles) |
| `timestamps` | Optional `M:SS Title` lines to split into tracks |
| `as_playlist`, `playlist_name` | Compilation mode: file each track per artist and write an M3U |

## Run it

From the phone: Wand → the `add_music` card, or a phone command Wand bins to it.
Over HTTP: `POST /api/wand/cast/add_music` with the params as the JSON body and the WizardOS bearer token.

Needs on the host: `wizardos`, `yt-dlp`, `ffmpeg`, `mutagen`.

Packaged from WizardOS (`~/wizard` on the Mini, commit `4e0ee17`).

## Integrate

### WizardOS

Copy `add_music.py` into `~/wizard/apps/wand/core/spells/`. Wand discovers every module there, so landing the
file is shipping the spell. Add its `extract` workload to `apps/wand/inference.yaml` so phone commands can fill
its params.
