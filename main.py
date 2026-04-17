import cv2
import numpy as np
import time

BLOCK_SIZE = 16

# -----------------------------
# Utils
# -----------------------------
def get_blocks(frame, block_size=BLOCK_SIZE):
    h, w, _ = frame.shape
    blocks = []

    for y in range(0, h, block_size):
        row = []
        for x in range(0, w, block_size):
            row.append(frame[y:y+block_size, x:x+block_size])
        blocks.append(row)

    return blocks


def rebuild_frame(blocks):
    rows = [np.hstack(r) for r in blocks]
    return np.vstack(rows)


def mse(b1, b2):
    return np.mean((b1.astype(float) - b2.astype(float)) ** 2)


# -----------------------------
# Motion estimation (your logic)
# -----------------------------
def find_best_match(block, blocks2, i, j, search_range=1):
    min_error = float("inf")
    best_vec = (0, 0)

    for di in range(-search_range, search_range+1):
        for dj in range(-search_range, search_range+1):
            ni, nj = i + di, j + dj

            if 0 <= ni < len(blocks2) and 0 <= nj < len(blocks2[0]):
                candidate = blocks2[ni][nj]

                if block.shape == candidate.shape:
                    err = mse(block, candidate)

                    if err < min_error:
                        min_error = err
                        best_vec = (di, dj)

    return min_error, best_vec


# -----------------------------
# Optical Flow Visualization
# -----------------------------
def draw_flow(frame, flow, step=16):
    h, w = frame.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x].astype(int)

            if abs(dx) < 1 and abs(dy) < 1:
                continue

            cv2.arrowedLine(frame, (x, y), (x+dx, y+dy), (0,255,0), 1)

    return frame


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def process_video(input_path):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    flow_out = cv2.VideoWriter("flow.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    recon_out = cv2.VideoWriter("reconstructed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    reconstructed_prev = prev.copy()
    recon_out.write(prev)

    threshold = 500

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------- Optical Flow --------
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
        vis = draw_flow(frame.copy(), flow)
        flow_out.write(vis)

        # -------- Encoding Decision --------
        error = np.mean((frame.astype(float) - reconstructed_prev.astype(float))**2)

        if error > threshold:
            # I-frame
            reconstructed = frame.copy()
        else:
            # P-frame
            blocks1 = get_blocks(frame)
            blocks2 = get_blocks(reconstructed_prev)

            new_blocks = []

            for i in range(len(blocks1)):
                row = []
                for j in range(len(blocks1[0])):

                    b1 = blocks1[i][j]
                    _, (dx, dy) = find_best_match(b1, blocks2, i, j)

                    ref = blocks2[i+dx][j+dy]
                    residual = b1.astype(np.int16) - ref.astype(np.int16)

                    new_block = ref.astype(np.int16) + residual
                    new_block = np.clip(new_block, 0, 255).astype(np.uint8)

                    row.append(new_block)

                new_blocks.append(row)

            reconstructed = rebuild_frame(new_blocks)

        recon_out.write(reconstructed)

        prev_gray = gray
        reconstructed_prev = reconstructed

    cap.release()
    flow_out.release()
    recon_out.release()
    cv2.destroyAllWindows()


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    start = time.time()
    process_video("videos/input.mp4")
    print("Time:", time.time() - start)