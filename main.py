import cv2
import numpy as np
import time

def load_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error opening video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, w, h


def draw_flow(frame, flow, step=16):
    h, w = frame.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x].astype(int)

            # draw arrow
            cv2.arrowedLine(
                frame,
                (x, y),
                (x + dx, y + dy),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

    return frame


def process_video(input_path, output_path):
    cap, fps, w, h = load_video(input_path)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,   # pyramid scale
            3,     # levels
            15,    # window size
            3,     # iterations
            5,     # poly_n
            1.2,   # poly_sigma
            0
        )

        # draw motion vectors
        vis = frame.copy()
        vis = draw_flow(vis, flow, step=16)

        out.write(vis)

        cv2.imshow("Optical Flow", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        prev_gray = gray

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start = time.time()
    process_video("videos/input.mp4", "output_flow.mp4")
    end = time.time()
    print("Time: ", end - start)