---
name: yt-transcribe
description: A YouTube video's transcript as timestamped JSON, from its captions or whisper as a fallback.
---

# yt-transcribe

When someone wants the text of a YouTube video, to read, search or quote.

One file, `yt_transcribe.py`. It does not import WizardOS.

## Params

| Param | Meaning |
|---|---|
| `url` | YouTube video URL, required |
| `--lang` | Caption language, default `en` |
| `--out` | JSON output path. Default: `~/.cache/yt-transcribe/transcripts/<id>.json` |

## Run it

```
python3 yt_transcribe.py 'https://www.youtube.com/watch?v=...'
```

The text goes to stdout. Progress goes to stderr.

Needs on the host: `yt-dlp` and `whisper-cli`. The first audio fallback downloads
`ggml-base.en.bin` into `~/.cache/yt-transcribe`. That file is public. It needs no token.
Use `ffmpeg` when it runs. On macOS, `afconvert` is the fallback when ffmpeg does not run.
