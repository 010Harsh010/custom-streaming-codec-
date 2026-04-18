(function () {
  const SEG_MAGIC = [0x53, 0x54, 0x52, 0x31]; // STR1
  const BLOCK_SIZE = 16;
  const Q = 8.0;
  const N = 8;

  const cosTable = Array.from({ length: N }, () => new Float32Array(N));
  for (let x = 0; x < N; x++) {
    for (let u = 0; u < N; u++) {
      cosTable[x][u] = Math.cos(((2 * x + 1) * u * Math.PI) / (2 * N));
    }
  }
  const cScale = new Float32Array(N);
  cScale[0] = 1 / Math.sqrt(2);
  for (let i = 1; i < N; i++) cScale[i] = 1;

  function clamp8(v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return v | 0;
  }

  function idct8x8(coeff64) {
    const out = new Float32Array(64);
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        let sum = 0.0;
        for (let v = 0; v < 8; v++) {
          for (let u = 0; u < 8; u++) {
            const idx = v * 8 + u;
            sum += cScale[u] * cScale[v] * coeff64[idx] * cosTable[x][u] * cosTable[y][v];
          }
        }
        out[y * 8 + x] = 0.25 * sum;
      }
    }
    return out;
  }

  function decodeResidualGray(coeff256) {
    const gray = new Float32Array(16 * 16);
    for (let tile = 0; tile < 4; tile++) {
      const tx = (tile % 2) * 8;
      const ty = ((tile / 2) | 0) * 8;
      const coeff64 = new Float32Array(64);
      const base = tile * 64;
      for (let i = 0; i < 64; i++) coeff64[i] = coeff256[base + i] * Q;
      const rec = idct8x8(coeff64);
      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
          gray[(ty + y) * 16 + (tx + x)] = rec[y * 8 + x] + 128.0;
        }
      }
    }
    return gray;
  }

  function createFrame(width, height) {
    return { width, height, data: new Uint8ClampedArray(width * height * 4) };
  }

  function copyFrame(src) {
    const f = createFrame(src.width, src.height);
    f.data.set(src.data);
    return f;
  }

  async function jpegToFrame(jpegBytes) {
    const blob = new Blob([jpegBytes], { type: "image/jpeg" });
    const bmp = await createImageBitmap(blob);
    const canvas = jpegToFrame._canvas || (jpegToFrame._canvas = document.createElement("canvas"));
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    canvas.width = bmp.width;
    canvas.height = bmp.height;
    ctx.drawImage(bmp, 0, 0);
    const image = ctx.getImageData(0, 0, bmp.width, bmp.height);
    return { width: bmp.width, height: bmp.height, data: image.data };
  }

  function getMacroblock(frame, mbRow, mbCol) {
    const out = new Uint8ClampedArray(BLOCK_SIZE * BLOCK_SIZE * 4);
    const startY = mbRow * BLOCK_SIZE;
    const startX = mbCol * BLOCK_SIZE;
    for (let y = 0; y < BLOCK_SIZE; y++) {
      const srcRow = (startY + y) * frame.width * 4;
      const dstRow = y * BLOCK_SIZE * 4;
      for (let x = 0; x < BLOCK_SIZE; x++) {
        const si = srcRow + (startX + x) * 4;
        const di = dstRow + x * 4;
        out[di] = frame.data[si];
        out[di + 1] = frame.data[si + 1];
        out[di + 2] = frame.data[si + 2];
        out[di + 3] = 255;
      }
    }
    return out;
  }

  function putMacroblock(frame, mbRow, mbCol, block) {
    const startY = mbRow * BLOCK_SIZE;
    const startX = mbCol * BLOCK_SIZE;
    for (let y = 0; y < BLOCK_SIZE; y++) {
      const dstRow = (startY + y) * frame.width * 4;
      const srcRow = y * BLOCK_SIZE * 4;
      for (let x = 0; x < BLOCK_SIZE; x++) {
        const si = srcRow + x * 4;
        const di = dstRow + (startX + x) * 4;
        frame.data[di] = block[si];
        frame.data[di + 1] = block[si + 1];
        frame.data[di + 2] = block[si + 2];
        frame.data[di + 3] = 255;
      }
    }
  }

  function applyResidualToBlock(refBlock, residualGray) {
    const out = new Uint8ClampedArray(refBlock.length);
    for (let y = 0; y < BLOCK_SIZE; y++) {
      for (let x = 0; x < BLOCK_SIZE; x++) {
        const p = y * BLOCK_SIZE + x;
        const r = residualGray[p];
        const i = p * 4;
        out[i] = clamp8(refBlock[i] + r);
        out[i + 1] = clamp8(refBlock[i + 1] + r);
        out[i + 2] = clamp8(refBlock[i + 2] + r);
        out[i + 3] = 255;
      }
    }
    return out;
  }

  function parseFramesFromPayload(payload) {
    const dv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    let off = 0;
    for (let i = 0; i < 4; i++) {
      if (payload[off + i] !== SEG_MAGIC[i]) throw new Error("bad segment magic");
    }
    off += 4;
    const version = dv.getUint8(off);
    off += 1;
    if (version !== 1) throw new Error("unsupported segment version");

    const frames = [];
    while (off < payload.length) {
      const frameType = dv.getUint8(off);
      off += 1;

      if (frameType === 0) {
        const n = dv.getUint32(off, true);
        off += 4;
        const jpeg = payload.slice(off, off + n);
        off += n;
        frames.push({ type: "I", jpeg });
      } else if (frameType === 1) {
        const nr = dv.getUint16(off, true);
        off += 2;
        const nc = dv.getUint16(off, true);
        off += 2;
        const blocks = new Array(nr);
        for (let i = 0; i < nr; i++) {
          blocks[i] = new Array(nc);
          for (let j = 0; j < nc; j++) {
            const dx = dv.getInt8(off);
            off += 1;
            const dy = dv.getInt8(off);
            off += 1;
            const coeff = new Int16Array(256);
            for (let k = 0; k < 256; k++) {
              coeff[k] = dv.getInt16(off, true);
              off += 2;
            }
            blocks[i][j] = { dx, dy, coeff };
          }
        }
        frames.push({ type: "P", nr, nc, blocks });
      } else {
        throw new Error(`unknown frame type byte ${frameType}`);
      }
    }
    return frames;
  }

  /**
   * @param {ArrayBuffer} arrayBuffer
   * @param {{ width: number, height: number, data: Uint8ClampedArray } | null} prevFrame
   * @param {{ onFrame?: (frame: { width: number, height: number, data: Uint8ClampedArray }) => void | Promise<void> }} [options]
   * If `onFrame` is set, each decoded frame is passed to it as soon as it is ready (streaming).
   */
  async function decodeSegment(arrayBuffer, prevFrame, options) {
    const onFrame = options && options.onFrame;
    const frames = onFrame ? null : [];

    const emit = async (f) => {
      const c = copyFrame(f);
      if (onFrame) await onFrame(c);
      else frames.push(c);
    };

    const compressed = new Uint8Array(arrayBuffer);
    const payload = window.pako.inflate(compressed);
    const frameUnits = parseFramesFromPayload(payload);

    let last = prevFrame ? copyFrame(prevFrame) : null;

    for (const unit of frameUnits) {
      if (unit.type === "I") {
        last = await jpegToFrame(unit.jpeg);
        await emit(last);
        continue;
      }

      if (!last) throw new Error("P-frame without reference");

      const out = createFrame(last.width, last.height);
      for (let i = 0; i < unit.nr; i++) {
        for (let j = 0; j < unit.nc; j++) {
          const b = unit.blocks[i][j];
          const refRow = i + b.dx;
          const refCol = j + b.dy;
          const refBlock = getMacroblock(last, refRow, refCol);
          const residualGray = decodeResidualGray(b.coeff);
          const recBlock = applyResidualToBlock(refBlock, residualGray);
          putMacroblock(out, i, j, recBlock);
        }
        if ((i & 3) === 3) await new Promise((r) => setTimeout(r, 0));
      }
      last = out;
      await emit(last);
    }

    return { frames: frames || [], lastFrame: last };
  }

  /**
   * Decodes segments in order and pushes each frame into a queue as soon as it is ready.
   * Playback uses nextFrame() only — no async gap between segment N and N+1 in the player.
   *
   * @param {{ fetchSegment: (path: string) => Promise<ArrayBuffer>, segmentPaths: string[], maxFrameBuffer?: number }} opts
   */
  function createBufferedFramePipeline({ fetchSegment, segmentPaths, maxFrameBuffer = 90 }) {
    const n = segmentPaths.length;
    const inflight = new Map();

    function ensureFetch(i) {
      if (i >= 0 && i < n && !inflight.has(i)) {
        inflight.set(i, fetchSegment(segmentPaths[i]));
      }
    }

    const frameQueue = [];
    let decodeError = null;
    let producerDone = false;
    let consumerWake = null;
    let producerWake = null;

    function wakeConsumer() {
      if (consumerWake) {
        const fn = consumerWake;
        consumerWake = null;
        fn();
      }
    }

    function wakeProducer() {
      if (producerWake) {
        const fn = producerWake;
        producerWake = null;
        fn();
      }
    }

    async function producer() {
      ensureFetch(0);
      ensureFetch(1);
      let prevLastFrame = null;
      try {
        for (let i = 0; i < n; i++) {
          ensureFetch(i + 2);
          const bytesPromise = inflight.get(i);
          if (!bytesPromise) throw new Error(`missing fetch for segment ${i}`);
          inflight.delete(i);
          const bytes = await bytesPromise;
          let frameInSeg = 0;
          const { lastFrame } = await decodeSegment(bytes, prevLastFrame, {
            onFrame: async (frame) => {
              while (frameQueue.length >= maxFrameBuffer) {
                await new Promise((r) => {
                  producerWake = r;
                });
              }
              frameQueue.push({
                frame,
                segmentIndex: i,
                frameInSegment: frameInSeg++,
                segmentPath: segmentPaths[i],
                segmentCount: n,
              });
              wakeConsumer();
            },
          });
          prevLastFrame = lastFrame;
        }
      } catch (err) {
        decodeError = err;
        wakeConsumer();
      } finally {
        producerDone = true;
        wakeConsumer();
      }
    }

    async function nextFrame() {
      while (frameQueue.length === 0 && !producerDone && !decodeError) {
        await new Promise((r) => {
          consumerWake = r;
        });
      }
      if (decodeError) throw decodeError;
      if (frameQueue.length === 0) return null;
      const item = frameQueue.shift();
      wakeProducer();
      return item;
    }

    function start() {
      producer();
    }

    function getQueueDepth() {
      return frameQueue.length;
    }

    return { start, nextFrame, getQueueDepth, segmentCount: n, maxFrameBuffer };
  }

  window.CustomTsDecoder = { decodeSegment, createBufferedFramePipeline };
})();
