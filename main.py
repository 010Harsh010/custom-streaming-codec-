import cv2
import numpy as np
import struct
import time
import os
import zlib

SEG_MAGIC = b"STR1"
SEG_VERSION = 1
COEFF_BYTES_PER_MB = 4 * 64 * 2  # four 8x8 tiles × 64 int16 coeffs

BLOCK_SIZE = 16

# -------------------------
# Padding
# -------------------------
def pad_frame(frame):
    h, w, _ = frame.shape
    new_h = (h + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    new_w = (w + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

    padded = np.zeros((new_h, new_w, 3), dtype=frame.dtype)
    padded[:h, :w] = frame
    return padded


# -------------------------
# Utils
# -------------------------
def get_blocks(frame):
    blocks = []
    h, w, _ = frame.shape

    for y in range(0, h, BLOCK_SIZE):
        row = []
        for x in range(0, w, BLOCK_SIZE):
            block = frame[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
            if block.shape == (BLOCK_SIZE, BLOCK_SIZE, 3):
                row.append(block)
        if row:
            blocks.append(row)

    return blocks


def rebuild_frame(blocks):
    return np.vstack([np.hstack(r) for r in blocks])


def mse(a, b):
    return np.mean((a.astype(np.float32) - b.astype(np.float32))**2)


# -------------------------
# DCT + Quantization
# -------------------------
Q = np.ones((8,8)) * 8   # lighter compression

def dct2(b):
    return cv2.dct(b.astype(np.float32))

def idct2(b):
    return cv2.idct(b.astype(np.float32))

def quantize(b):
    return np.round(b / Q)

def dequantize(b):
    return b * Q


# -------------------------
# Motion estimation
# -------------------------
def find_best_match(block, blocks2, i, j):
    best = (0,0)
    min_err = float("inf")

    for di in range(-1,2):
        for dj in range(-1,2):
            ni, nj = i+di, j+dj
            if 0<=ni<len(blocks2) and 0<=nj<len(blocks2[0]):
                cand = blocks2[ni][nj]
                if block.shape != cand.shape:
                    continue
                err = mse(block, cand)
                if err < min_err:
                    min_err = err
                    best = (di,dj)
    return best


# -------------------------
# Optical Flow
# -------------------------
def draw_flow(frame, flow):
    for y in range(0, frame.shape[0], 16):
        for x in range(0, frame.shape[1], 16):
            dx, dy = flow[y,x].astype(int)
            if abs(dx)+abs(dy) > 1:
                cv2.arrowedLine(frame, (x,y), (x+dx,y+dy), (0,255,0), 1)
    return frame


# -------------------------
# Residual Compression (FIXED)
# -------------------------
def encode_residual(residual):
    # Encode 16x16 residual as four 8x8 DCT blocks, quantize, and flatten to int16
    gray = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tiles = []

    for y in range(0, gray.shape[0], 8):
        for x in range(0, gray.shape[1], 8):
            block = gray[y:y+8, x:x+8]
            if block.shape != (8, 8):
                continue

            block = block - 128
            d = dct2(block)
            q = quantize(d)
            q_int = np.round(q).astype(np.int16)
            tiles.append(q_int.reshape(64))

    return np.stack(tiles, axis=0)


def decode_residual(coeffs, shape):
    h, w = shape
    res = np.zeros((h, w), dtype=np.float32)
    c = np.asarray(coeffs, dtype=np.int16)
    tile_idx = 0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = c[tile_idx].reshape(8, 8).astype(np.float32)
            tile_idx += 1
            d = dequantize(block)
            rec = idct2(d)
            rec = rec + 128
            res[y:y+8, x:x+8] = rec
    return res


# -------------------------
# MAIN PIPELINE
# -------------------------
def process_video(path):
    Info_logs = []
    cap = cv2.VideoCapture(path)

    ret, first = cap.read()
    if not ret or first is None:
        raise RuntimeError(f"could not read first frame from {path}")
    
    # Add padding for block processing 16*16
    first = pad_frame(first)

    h, w, _ = first.shape
    print(f"Data Size: {h*w*3*cap.get(cv2.CAP_PROP_FRAME_COUNT)/(1024*1024):.2f} MB (padded frames)")
    Info_logs.append(f"Data Size: {h*w*3*cap.get(cv2.CAP_PROP_FRAME_COUNT)/(1024*1024):.2f} MB (padded frames)")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"FPS: {fps}")
    Info_logs.append(f"FPS: {fps}")

    if fps <= 0:
        fps = 30

    os.makedirs("segments", exist_ok=True)

    flow_out = cv2.VideoWriter("flow.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    recon_out = cv2.VideoWriter("reconstructed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    prev = first
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    print("Gray Data Size:", h*w*cap.get(cv2.CAP_PROP_FRAME_COUNT) / (1024*1024), "MB")
    
    Info_logs.append(f"Gray Data Size: {h*w*cap.get(cv2.CAP_PROP_FRAME_COUNT)/(1024*1024):.2f} MB")

    reconstructed_prev = prev.copy()
    recon_out.write(prev)

    encoded = []
    log = []

    # Always start stream with an I-frame.
    ok, first_buf = cv2.imencode(".jpg", first, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError("failed to encode first frame as I-frame")
    encoded.append({"type": "I", "frame": first_buf.tobytes()})
    log.append((0, "I", 0.0))

    idx = 1
    threshold = 500
    
    No_of_I_frames = 1
    No_of_P_frames = 0
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = pad_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
        flow_out.write(draw_flow(frame.copy(), flow))

        error = mse(frame, reconstructed_prev)

        # -------- I FRAME --------
        if error > threshold:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY,70])
            encoded.append({"type":"I", "frame":buf.tobytes()})
            reconstructed = frame.copy()
            ftype = "I"
            No_of_I_frames += 1

        # -------- P FRAME --------
        else:
            b1 = get_blocks(frame)
            b2 = get_blocks(reconstructed_prev)

            enc_blocks = []
            new_blocks = []

            for i in range(len(b1)):
                row_enc, row_new = [], []

                for j in range(len(b1[0])):
                    dx, dy = find_best_match(b1[i][j], b2, i, j)
                    ref = b2[i+dx][j+dy]

                    residual = b1[i][j].astype(np.float32) - ref.astype(np.float32)

                    comp = encode_residual(residual)
                    rec_res = decode_residual(comp, ref.shape[:2])

                    new = ref.astype(np.float32)

                    # apply residual to all channels
                    for c in range(3):
                        new[:,:,c] += rec_res

                    new = np.clip(new, 0, 255).astype(np.uint8)

                    row_enc.append((dx,dy,comp))
                    row_new.append(new)

                enc_blocks.append(row_enc)
                new_blocks.append(row_new)

            encoded.append({"type":"P", "blocks":enc_blocks})
            reconstructed = rebuild_frame(new_blocks)
            ftype = "P"
            No_of_P_frames += 1

        log.append((idx, ftype, idx/fps))
        recon_out.write(reconstructed)

        reconstructed_prev = reconstructed
        prev_gray = gray
        idx += 1

    cap.release()
    flow_out.release()
    recon_out.release()
    
    Info_logs.append(f"Total Frames: {idx}")
    Info_logs.append(f"I-Frames: {No_of_I_frames}")
    Info_logs.append(f"P-Frames: {No_of_P_frames}")
    
    Info_logs.append(f"Total Size: {sum(len(e['frame']) if e['type']=='I' else len(e['blocks'])*BLOCK_SIZE*BLOCK_SIZE*2 for e in encoded)/(1024*1024):.2f} MB")

    return encoded, log, fps, Info_logs


# -------------------------
# Segment blob (compact binary, zlib in .ts file)
# -------------------------
def serialize_encoded(frames):
    parts = [SEG_MAGIC, struct.pack("<B", SEG_VERSION)]
    for fr in frames:
        if fr["type"] == "I":
            jpeg = fr["frame"]
            parts.append(struct.pack("<BI", 0, len(jpeg)))
            parts.append(jpeg)
        else:
            blocks = fr["blocks"]
            nr, nc = len(blocks), len(blocks[0])
            parts.append(struct.pack("<BHH", 1, nr, nc))
            for i in range(nr):
                for j in range(nc):
                    dx, dy, coeff = blocks[i][j]
                    c = np.asarray(coeff, dtype=np.int16)
                    if c.size != 256:
                        raise ValueError(f"expected 256 int16 coeffs per macroblock, got {c.size}")
                    parts.append(struct.pack("<bb", int(dx), int(dy)))
                    parts.append(c.tobytes())
    return b"".join(parts)


def deserialize_encoded(blob):
    if len(blob) < 5 or blob[:4] != SEG_MAGIC:
        raise ValueError("invalid segment: bad magic")
    ver = blob[4]
    if ver != SEG_VERSION:
        raise ValueError(f"unsupported segment version {ver}")
    off = 5
    out = []
    while off < len(blob):
        t = blob[off]
        off += 1
        if t == 0:
            n = struct.unpack_from("<I", blob, off)[0]
            off += 4
            jpeg = blob[off : off + n]
            off += n
            out.append({"type": "I", "frame": jpeg})
        elif t == 1:
            nr, nc = struct.unpack_from("<HH", blob, off)
            off += 4
            blocks = []
            for _i in range(nr):
                row = []
                for _j in range(nc):
                    dx, dy = struct.unpack_from("<bb", blob, off)
                    off += 2
                    raw = blob[off : off + COEFF_BYTES_PER_MB]
                    off += COEFF_BYTES_PER_MB
                    coeff = np.frombuffer(raw, dtype=np.int16).copy().reshape(4, 64)
                    row.append((dx, dy, coeff))
                blocks.append(row)
            out.append({"type": "P", "blocks": blocks})
        else:
            raise ValueError(f"unknown frame type byte {t}")
    return out


# -------------------------
# SEGMENTATION (.ts extension, custom payload)
# -------------------------
def segment_video(encoded, log, fps, Info_logs):
    points = [0]
    last = 0

    for i,t,ts in log:
        if ts - last >= 5 and t == "I":
            points.append(i)
            last = ts

    for i in range(len(points)):
        s = points[i]
        e = points[i+1] if i+1 < len(points) else len(encoded)

        payload = serialize_encoded(encoded[s:e])
        data = zlib.compress(payload, 9)

        Info_logs.append(f"Segment {i}: Frames {s} to {e-1}, Size: {len(data)/(1024*1024):.2f} MB")
        
        with open(f"segments/segment_{i}.ts","wb") as f:
            f.write(data)

    with open("index.m3u8","w") as f:
        f.write("#EXTM3U\n")
        fps_d = max(float(fps), 1e-6)
        for i in range(len(points)):
            s = points[i]
            e = points[i + 1] if i + 1 < len(points) else len(encoded)
            dur = (e - s) / fps_d
            f.write(f"#EXTINF:{dur:.6f},\nsegments/segment_{i}.ts\n")


# -------------------------
# DECODER
# -------------------------
def play():
    with open("index.m3u8") as f:
        segs = [l.strip() for l in f if l.strip().endswith(".ts")]

    prev_decoded = None
    for s in segs:
        data = deserialize_encoded(zlib.decompress(open(s, "rb").read()))

        frames = []

        for d in data:
            if d["type"] == "I":
                img = np.frombuffer(d["frame"], np.uint8)
                frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
                frames.append(frame)
                prev_decoded = frame

            else:
                prev = frames[-1] if frames else prev_decoded
                if prev is None:
                    raise RuntimeError(
                        "segment starts with P-frame and no reference frame is available"
                    )
                b2 = get_blocks(prev)

                new_blocks = []

                for i in range(len(d["blocks"])):
                    row = []
                    for j in range(len(d["blocks"][0])):
                        dx,dy,comp = d["blocks"][i][j]

                        ref = b2[i+dx][j+dy]
                        res = decode_residual(comp, ref.shape[:2])

                        new = ref.astype(np.float32)

                        for c in range(3):
                            new[:,:,c] += res

                        new = np.clip(new, 0, 255).astype(np.uint8)

                        row.append(new)

                    new_blocks.append(row)

                rec = rebuild_frame(new_blocks)
                frames.append(rec)
                prev_decoded = rec

        for f in frames:
            cv2.imshow("Stream", f)
            if cv2.waitKey(30) == 27:
                cv2.destroyAllWindows() 
                return
            
    cv2.destroyAllWindows() 
    return 


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    start = time.time()

    encoded, log, fps, Info_logs = process_video("videos/input.mp4")

    with open("frame_log.txt","w") as f:
        for i,t,ts in log:
            f.write(f"Frame {i}: {t}, Time: {ts:.2f}s\n")

    segment_video(encoded, log, fps, Info_logs)
    
    with open("info_log.txt","w") as f:
        for line in Info_logs:
            f.write(line + "\n")

    print("Done Encoding + Segmentation")

    # play()

    print("Time:", time.time() - start)
