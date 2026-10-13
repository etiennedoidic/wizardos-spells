---
name: yt-transcribe
description: A YouTube video's transcript as timestamped JSON, from its captions or whisper as a fallback, filed as a Grimoire note.
---

# yt-transcribe

When someone wants the text of a YouTube video, to read, search or quote.

A Wand spell: one file, `yt_transcribe.py`, declared with `@spell("yt_transcribe", …)`. Wand renders its form from
the params model and runs it in the worker as a `wand.cast` job, streaming progress lines to the phone.

## Params

| Param | Meaning |
|---|---|
| `url` | YouTube video URL, required |
| `lang` | Caption language, default `en` |

## Run it

From the phone: Wand → the `yt_transcribe` card, or a phone command Wand bins to it.
Over HTTP: `POST /api/wand/cast/yt_transcribe` with the params as the JSON body and the WizardOS bearer token.

Needs on the host: `wizardos`, `yt-dlp`, `whisper-cli`.

Packaged from WizardOS (`~/wizard` on the Mini, commit `4e0ee17`).

## Integrate

### WizardOS

Copy `yt_transcribe.py` into `~/wizard/apps/wand/core/spells/`. Wand discovers every module there, so landing the
file is shipping the spell. Add its `extract` workload to `apps/wand/inference.yaml` so phone commands can fill
its params.
