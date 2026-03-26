"""
5-point gaze calibration screen.
Maps raw MediaPipe iris positions → frame pixel coordinates.
Run once before vision_node.py and save the homography matrix.
"""

import os

import cv2
import mediapipe as mp
import numpy as np

CALIB_POINTS = [
    (0.1, 0.1),
    (0.9, 0.1),
    (0.5, 0.5),
    (0.1, 0.9),
    (0.9, 0.9),
]
WINDOW = "Calibration — look at each dot and press SPACE"
W, H = 1280, 720


def main():
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
    )

    raw_iris = []
    screen_pts = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    for px, py in CALIB_POINTS:
        sx, sy = int(px * W), int(py * H)
        print(f"Look at the dot at ({sx}, {sy}) then press SPACE")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis = frame.copy()
            cv2.circle(vis, (sx, sy), 20, (0, 255, 0), -1)
            cv2.imshow(WINDOW, vis)
            key = cv2.waitKey(1)
            if key == ord(" "):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    lm = result.multi_face_landmarks[0].landmark
                    ix = (lm[468].x + lm[473].x) / 2 * W
                    iy = (lm[468].y + lm[473].y) / 2 * H
                    raw_iris.append([ix, iy])
                    screen_pts.append([sx, sy])
                    print(f"  Recorded iris=({ix:.1f}, {iy:.1f})")
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(raw_iris) < 4:
        print("Need at least 4 calibration points with a visible face; exiting.")
        raise SystemExit(1)

    h_mat, _ = cv2.findHomography(
        np.array(raw_iris, dtype=np.float32),
        np.array(screen_pts, dtype=np.float32),
    )

    cal_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(cal_dir, exist_ok=True)
    out_path = os.path.join(cal_dir, "homography.npy")
    np.save(out_path, h_mat)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
