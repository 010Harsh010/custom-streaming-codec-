import cv2
import numpy as np
import pickle
import time
import os

BLOCK_SIZE = 16

# -------------------------
# Utils
# -------------------------
def get_blocks(frame):
    h, w, _ = frame.shape
    blocks = []
    for y in range(0, h, BLOCK_SIZE):
        row = []
        for x in range(0, w, BLOCK_SIZE):
            row.append(frame[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE])
        blocks.append(row)
    return blocks


def rebuild_frame(blocks):
    rows = [np.hstack(r) for r in blocks]
    return np.vstack(rows)


def mse(b1, b2):
    return np.mean((b1.astype(float) - b2.astype(float))**2)


# -------------------------
# Motion estimation
# -------------------------
def find_best_match(block, blocks2, i, j, search_range=1):
    min_err = float("inf")
    best = (0, 0)

    for di in range(-search_range, search_range+1):
        for dj in range(-search_range, search_range+1):
            ni, nj = i+di, j+dj
            if 0 <= ni < len(blocks2) and 0 <= nj < len(blocks2[0]):
                candidate = blocks2[ni][nj]
                if block.shape == candidate.shape:
                    err = mse(block, candidate)
                    if err < min_err:
                        min_err = err
                        best = (di, dj)

    return min_err, best


# -------------------------
# Optical Flow
# -------------------------
def draw_flow(frame, flow, step=16):
    h, w = frame.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x].astype(int)
            if abs(dx) < 1 and abs(dy) < 1:
                continue
            cv2.arrowedLine(frame, (x,y), (x+dx, y+dy), (0,255,0), 1)
    return frame


# -------------------------
# MAIN PIPELINE
# -------------------------
def process_video(input_path):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("segments", exist_ok=True)

    flow_out = cv2.VideoWriter("flow.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    recon_out = cv2.VideoWriter("reconstructed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    reconstructed_prev = prev.copy()
    recon_out.write(prev)

    encoded_video = []
    frame_log = []
    reconstructed_frames = [prev]

    threshold = 500
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
        vis = draw_flow(frame.copy(), flow)
        flow_out.write(vis)

        error = np.mean((frame.astype(float) - reconstructed_prev.astype(float))**2)

        # -------- I FRAME --------
        if error > threshold:
            frame_type = "I"
            encoded_video.append({
                "type": "I",
                "frame": frame
            })
            reconstructed = frame.copy()

        # -------- P FRAME --------
        else:
            frame_type = "P"

            blocks1 = get_blocks(frame)
            blocks2 = get_blocks(reconstructed_prev)

            encoded_blocks = []
            new_blocks = []

            for i in range(len(blocks1)):
                row_enc = []
                row_new = []

                for j in range(len(blocks1[0])):
                    b1 = blocks1[i][j]

                    _, (dx, dy) = find_best_match(b1, blocks2, i, j)

                    ref = blocks2[i+dx][j+dy]
                    residual = b1.astype(np.int16) - ref.astype(np.int16)

                    new_block = ref.astype(np.int16) + residual
                    new_block = np.clip(new_block, 0, 255).astype(np.uint8)

                    row_enc.append((dx, dy, residual))
                    row_new.append(new_block)

                encoded_blocks.append(row_enc)
                new_blocks.append(row_new)

            encoded_video.append({
                "type": "P",
                "blocks": encoded_blocks
            })

            reconstructed = rebuild_frame(new_blocks)

        # Save
        frame_log.append((frame_idx, frame_type, time_sec))
        reconstructed_frames.append(reconstructed)
        recon_out.write(reconstructed)

        prev_gray = gray
        reconstructed_prev = reconstructed
        frame_idx += 1

    cap.release()
    flow_out.release()
    recon_out.release()

    return encoded_video, frame_log, reconstructed_frames, fps, w, h


# -------------------------
# SEGMENTATION
# -------------------------
def get_segment_points(frame_log, fps):
    points = [0]
    last = 0

    for idx, t, ts in frame_log:
        if ts - last >= 5 and t == "I":
            points.append(idx)
            last = ts

    return points


def save_segments(encoded_video, points):
    for i in range(len(points)):
        start = points[i]
        end = points[i+1] if i+1 < len(points) else len(encoded_video)

        segment = encoded_video[start:end]

        with open(f"segments/segment_{i}.seg", "wb") as f:
            pickle.dump(segment, f)


def create_index(points, fps):
    with open("index.m3u8", "w") as f:
        f.write("#EXTM3U\n")

        for i in range(len(points)):
            start = points[i]
            end = points[i+1] if i+1 < len(points) else start + fps*5

            duration = (end - start)/fps

            f.write(f"#EXTINF:{duration:.2f},\n")
            f.write(f"segments/segment_{i}.seg\n")


# -------------------------
# DECODER
# -------------------------
def decode_segment(segment):
    frames = []

    for i, data in enumerate(segment):
        if data["type"] == "I":
            frames.append(data["frame"])

        else:
            prev = frames[-1]
            prev_blocks = get_blocks(prev)

            new_blocks = []

            for r in range(len(data["blocks"])):
                row = []
                for c in range(len(data["blocks"][0])):
                    dx, dy, residual = data["blocks"][r][c]

                    ref = prev_blocks[r+dx][c+dy]
                    block = ref.astype(np.int16) + residual
                    block = np.clip(block, 0, 255).astype(np.uint8)

                    row.append(block)

                new_blocks.append(row)

            frames.append(rebuild_frame(new_blocks))

    return frames


def play_stream():
    with open("index.m3u8") as f:
        lines = f.readlines()

    segments = [l.strip() for l in lines if l.endswith(".seg")]

    for seg in segments:
        with open(seg, "rb") as f:
            segment = pickle.load(f)

        frames = decode_segment(segment)

        for frame in frames:
            cv2.imshow("Stream", frame)
            if cv2.waitKey(30) == 27:
                return


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    start = time.time()

    encoded_video, frame_log, reconstructed_frames, fps, w, h = process_video("videos/input.mp4")

    with open("frame_log.txt", "w") as f:
        for idx, t, ts in frame_log:
            f.write(f"Frame {idx}: {t}, Time: {ts:.2f}s\n")

    points = get_segment_points(frame_log, fps)
    save_segments(encoded_video, points)
    create_index(points, fps)

    print("Encoding + Segmentation Done")

    # Play stream
    play_stream()

    print("Time:", time.time() - start)