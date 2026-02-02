# musicvid

Textual TUI that converts a YouTube (or local) video into an ASCII / Braille "music video":

1. download (if URL)
2. extract frames every N seconds
3. convert each frame to ASCII or Braille text
4. render text to PNG frames
5. encode PNGs back to MP4 and mux original audio
6. encode PNGs back to MP4 and mux original audio

## Requirements

- `ffmpeg` on PATH
- Python 3.10+

Python deps:

```bash
pip install -r requirements.txt
```

For reliable YouTube downloads, install a JS runtime (Node) so yt-dlp can solve modern YouTube JS challenges:

```bash
sudo apt-get update
sudo apt-get install -y nodejs npm
```

## Run

```bash
python3 ascii_mv_tui.py
```

## Making It “Continuous”

Set **FPS** to something like `5` (lighter) or `10` (smoother). The old default of 5 seconds per frame was a slideshow.

Preview text updates inside the TUI as frames render.
# visualizerdot
