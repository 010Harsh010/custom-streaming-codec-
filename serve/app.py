from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

ROOT_DIR = Path(__file__).resolve().parents[1]
SERVE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Adaptive Segment Player")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PLAYER_HTML


@app.get("/stream/{playlist_path:path}", response_class=PlainTextResponse)
def stream_playlist(playlist_path: str) -> str:
    # Keep reads inside project root.
    full = (ROOT_DIR / playlist_path).resolve()
    if ROOT_DIR not in full.parents and full != ROOT_DIR:
        raise HTTPException(status_code=400, detail="invalid playlist path")
    if not full.exists():
        raise HTTPException(status_code=404, detail="playlist not found")
    return full.read_text(encoding="utf-8")


@app.get("/media/{asset_path:path}")
def media(asset_path: str):
    full = (ROOT_DIR / asset_path).resolve()
    if ROOT_DIR not in full.parents and full != ROOT_DIR:
        raise HTTPException(status_code=400, detail="invalid media path")
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="media not found")
    return FileResponse(full, media_type="application/octet-stream")


@app.get("/decode.js")
def decode_js():
    return FileResponse(SERVE_DIR / "decode.js", media_type="application/javascript")


PLAYER_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Adaptive segment player</title>
    <style>
      :root {
        color-scheme: dark;
        font-family: system-ui, Segoe UI, Arial, sans-serif;
      }
      body {
        margin: 0;
        background: #12131a;
        color: #e8e8ef;
      }
      main {
        max-width: 980px;
        margin: 0 auto;
        padding: 18px;
      }
      h1 {
        margin: 0 0 8px;
        font-size: 1.35rem;
        font-weight: 600;
      }
      .hint {
        margin: 0 0 12px;
        color: #a9adbc;
        font-size: 0.92rem;
      }
      .controls {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: end;
      }
      label {
        display: flex;
        flex-direction: column;
        gap: 4px;
        font-size: 0.85rem;
        color: #c5c8d4;
      }
      input {
        min-width: 240px;
        padding: 7px 9px;
        border-radius: 6px;
        border: 1px solid #2c2f3d;
        background: #1a1c27;
        color: #e8e8ef;
      }
      button {
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #3d4260;
        background: #2a2f48;
        color: #e8e8ef;
        cursor: pointer;
      }
      button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      .stats {
        margin: 12px 0;
        display: grid;
        gap: 5px;
        color: #c5c8d4;
        font-size: 0.88rem;
      }
      canvas {
        width: 100%;
        max-width: 960px;
        height: auto;
        background: #000;
        border-radius: 8px;
        border: 1px solid #2c2f3d;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>Adaptive segment player</h1>
      <p class="hint">Frames are queued as they decode (no gap between segments); playback uses a steady clock with a short buffer.</p>
      <section class="controls">
        <label>Server base URL
          <input id="baseUrl" value="http://127.0.0.1:8765" />
        </label>
        <label>Playlist path
          <input id="playlistPath" value="index.m3u8" />
        </label>
        <button id="playBtn" type="button">Play</button>
        <button id="stopBtn" type="button" disabled>Stop</button>
      </section>
      <section class="stats">
        <div>Status: <span id="status">idle</span></div>
        <div>Segment: <span id="segInfo">-</span></div>
        <div>Frames: <span id="frameInfo">-</span></div>
        <div>Decode: <span id="decodeInfo">-</span></div>
        <div>Buffer: <span id="bufferInfo">-</span></div>
      </section>
      <canvas id="screen" width="960" height="540"></canvas>
    </main>

    <script src="https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js"></script>
    <script src="/decode.js"></script>
    <script>
      const canvas = document.getElementById("screen");
      const ctx = canvas.getContext("2d");
      const statusEl = document.getElementById("status");
      const segEl = document.getElementById("segInfo");
      const frameEl = document.getElementById("frameInfo");
      const decodeEl = document.getElementById("decodeInfo");
      const bufferEl = document.getElementById("bufferInfo");
      const playBtn = document.getElementById("playBtn");
      const stopBtn = document.getElementById("stopBtn");
      let stopFlag = false;

      const setStatus = (txt) => { statusEl.textContent = txt; };
      const sleep = (ms) => new Promise(r => setTimeout(r, ms));

      async function readPlaylist(baseUrl, playlistPath) {
        const resp = await fetch(`${baseUrl}/stream/${playlistPath}`);
        if (!resp.ok) throw new Error("cannot load playlist");
        const text = await resp.text();
        return text
          .split("\\n")
          .map(s => s.trim())
          .filter(s => s.endsWith(".ts"));
      }

      async function fetchSegment(baseUrl, segPath) {
        const resp = await fetch(`${baseUrl}/media/${segPath}`);
        if (!resp.ok) throw new Error(`fetch failed for ${segPath}`);
        return await resp.arrayBuffer();
      }

      function drawFrame(frame) {
        if (canvas.width !== frame.width || canvas.height !== frame.height) {
          canvas.width = frame.width;
          canvas.height = frame.height;
        }
        const imgData = new ImageData(frame.data, frame.width, frame.height);
        ctx.putImageData(imgData, 0, 0);
      }

      async function play() {
        stopFlag = false;
        playBtn.disabled = true;
        stopBtn.disabled = false;
        segEl.textContent = "-";
        frameEl.textContent = "-";
        decodeEl.textContent = "-";
        bufferEl.textContent = "-";

        const baseUrl = document.getElementById("baseUrl").value.trim();
        const playlistPath = document.getElementById("playlistPath").value.trim();
        setStatus("loading playlist...");

        try {
          const segs = await readPlaylist(baseUrl, playlistPath);
          if (segs.length === 0) throw new Error("no .ts segments in playlist");

          const pipeline = window.CustomTsDecoder.createBufferedFramePipeline({
            segmentPaths: segs,
            fetchSegment: (path) => fetchSegment(baseUrl, path),
            maxFrameBuffer: 90,
          });
          pipeline.start();

          const FRAME_MS = 1000 / 30;
          const MIN_BUFFERED_FRAMES = 12;
          const PRIME_MS = 5000;
          const tPrime = performance.now();
          while (!stopFlag && performance.now() - tPrime < PRIME_MS) {
            const d = pipeline.getQueueDepth();
            if (d >= MIN_BUFFERED_FRAMES) break;
            if (d >= 1 && performance.now() - tPrime > 280) break;
            await sleep(6);
          }

          let nextPresent = performance.now();
          decodeEl.textContent = "streaming";

          while (true) {
            if (stopFlag) break;
            const item = await pipeline.nextFrame();
            if (!item) break;

            segEl.textContent = `${item.segmentIndex + 1}/${item.segmentCount} (${item.segmentPath})`;
            frameEl.textContent = `#${item.frameInSegment + 1} in segment`;
            bufferEl.textContent = `frames buffered: ${pipeline.getQueueDepth()} (cap ${pipeline.maxFrameBuffer})`;
            setStatus("playing");

            drawFrame(item.frame);

            const now = performance.now();
            let wait = nextPresent - now;
            nextPresent += FRAME_MS;
            if (wait < -FRAME_MS * 3) {
              nextPresent = now + FRAME_MS;
              wait = 0;
            } else if (wait < 0) {
              wait = 0;
            }
            if (wait > 0) await sleep(wait);
          }
          if (!stopFlag) setStatus("done");
        } catch (err) {
          setStatus("error: " + err.message);
        } finally {
          playBtn.disabled = false;
          stopBtn.disabled = true;
        }
      }

      playBtn.onclick = () => { play(); };
      stopBtn.onclick = () => { stopFlag = true; setStatus("stopped"); };
    </script>
  </body>
</html>
"""
