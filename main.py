import cv2
import numpy as np
import pickle
import time
import os
import zlib

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
# RLE
# -------------------------
def rle_encode(arr):
    out = []
    count = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            count += 1
        else:
            out.append((arr[i-1], count))
            count = 1
    out.append((arr[-1], count))
    return out

def rle_decode(arr):
    out = []
    for v, c in arr:
        out.extend([v]*c)
    return out


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
    # keep float, DO NOT cast to uint8
    gray = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY).astype(np.float32)

    compressed = []

    for y in range(0, gray.shape[0], 8):
        for x in range(0, gray.shape[1], 8):
            block = gray[y:y+8, x:x+8]
            if block.shape != (8,8):
                continue

            # level shift
            block = block - 128

            d = dct2(block)
            q = quantize(d)
            rle = rle_encode(q.flatten())

            compressed.append(rle)

    return compressed


def decode_residual(comp, shape):
    h, w = shape
    res = np.zeros((h,w), dtype=np.float32)

    idx = 0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if idx >= len(comp):
                break

            arr = rle_decode(comp[idx])
            idx += 1

            block = np.array(arr).reshape(8,8)

            d = dequantize(block)
            rec = idct2(d)

            rec = rec + 128   # reverse level shift

            res[y:y+8, x:x+8] = rec

    return res


# -------------------------
# MAIN PIPELINE
# -------------------------
def process_video(path):
    cap = cv2.VideoCapture(path)

    ret, first = cap.read()
    first = pad_frame(first)

    h, w, _ = first.shape
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs("segments", exist_ok=True)

    flow_out = cv2.VideoWriter("flow.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    recon_out = cv2.VideoWriter("reconstructed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    prev = first
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    reconstructed_prev = prev.copy()
    recon_out.write(prev)

    encoded = []
    log = []

    idx = 0
    threshold = 500

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

        log.append((idx, ftype, idx/fps))
        recon_out.write(reconstructed)

        reconstructed_prev = reconstructed
        prev_gray = gray
        idx += 1

    cap.release()
    flow_out.release()
    recon_out.release()

    return encoded, log, fps


# -------------------------
# SEGMENTATION
# -------------------------
def segment_video(encoded, log, fps):
    points = [0]
    last = 0

    for i,t,ts in log:
        if ts - last >= 5 and t == "I":
            points.append(i)
            last = ts

    for i in range(len(points)):
        s = points[i]
        e = points[i+1] if i+1 < len(points) else len(encoded)

        data = zlib.compress(pickle.dumps(encoded[s:e]), 9)

        with open(f"segments/segment_{i}.seg","wb") as f:
            f.write(data)

    with open("index.m3u8","w") as f:
        f.write("#EXTM3U\n")
        for i in range(len(points)):
            f.write(f"#EXTINF:5,\nsegments/segment_{i}.seg\n")


# -------------------------
# DECODER
# -------------------------
def play():
    with open("index.m3u8") as f:
        segs = [l.strip() for l in f if l.endswith(".seg")]

    for s in segs:
        data = pickle.loads(zlib.decompress(open(s,"rb").read()))

        frames = []

        for d in data:
            if d["type"] == "I":
                img = np.frombuffer(d["frame"], np.uint8)
                frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
                frames.append(frame)

            else:
                prev = frames[-1]
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

                frames.append(rebuild_frame(new_blocks))

        for f in frames:
            cv2.imshow("Stream", f)
            if cv2.waitKey(30) == 27:
                return


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    start = time.time()

    encoded, log, fps = process_video("videos/input.mp4")

    with open("frame_log.txt","w") as f:
        for i,t,ts in log:
            f.write(f"Frame {i}: {t}, Time: {ts:.2f}s\n")

    segment_video(encoded, log, fps)

    print("Done Encoding + Segmentation")

    play()

    print("Time:", time.time() - start)