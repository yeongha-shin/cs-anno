"""Quick visual check: draw auto-labels + the geometric cues that drive them.

All on-image text is English (OpenCV's bitmap font cannot render Hangul).
Every reference point is labeled with what it means (chin, wrist, ...) so the
heuristic decisions are inspectable at a glance.
"""
import sys, cv2, numpy as np
from pipeline.features import FeatureExtractor
from pipeline.labelers import LABELERS, IPAD_Y

ex = FeatureExtractor()

# English dimension titles for the overlay (SCHEMA titles are Korean)
DIM_TITLE = {"attention": "ATTENTION", "emotion": "EMOTION",
             "action": "ACTION", "posture": "POSTURE"}

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (240, 240, 240)
RED = (60, 60, 255)


def _pt(p, w, h):
    return (int(p[0] * w), int(p[1] * h))


def _dot(img, p, w, h, color, label, lblcolor=WHITE):
    x, y = _pt(p, w, h)
    cv2.circle(img, (x, y), 7, color, -1)
    cv2.circle(img, (x, y), 8, (0, 0, 0), 1)
    cv2.putText(img, label, (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3)               # black outline
    cv2.putText(img, label, (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, lblcolor, 1)


def annotate(path, out):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    f = ex.extract(rgb)
    h, w = bgr.shape[:2]
    labels = {dim: fn(f) for dim, fn in LABELERS.items()}

    # iPad zone: hand below this line -> action "touching"
    yline = int(IPAD_Y * h)
    cv2.line(bgr, (0, yline), (w, yline), (0, 140, 255), 2)
    cv2.putText(bgr, "iPad zone (hand below = touching)", (15, yline - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    # ---- geometric cues ----
    if f.get("face_detected"):
        # face bbox (+10% margin) = the region used for face_touching
        if "face_bbox" in f:
            x0, y0, x1, y1 = f["face_bbox"]
            mx, my = (x1 - x0) * 0.10, (y1 - y0) * 0.10
            cv2.rectangle(bgr, (int((x0 - mx) * w), int((y0 - my) * h)),
                          (int((x1 + mx) * w), int((y1 + my) * h)), (0, 200, 0), 2)
            cv2.putText(bgr, "face zone (hand inside = face_touching)",
                        (int((x0 - mx) * w), int((y0 - my) * h) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        if "chin" in f:
            _dot(bgr, f["chin"], w, h, MAGENTA, "chin (face #152)")
        if "face_cx" in f:
            _dot(bgr, (f["face_cx"], f["face_cy"]), w, h, CYAN, "face_center")

        # hands: label fingertip + wrist, and draw the hand->chin line
        # (that distance, normalized by face height, drives chin_rest/touching)
        chin = f.get("chin")
        for i, hand in enumerate(f.get("hands", [])):
            _dot(bgr, hand["index"], w, h, YELLOW, f"hand{i} index_tip")
            _dot(bgr, hand["wrist"], w, h, YELLOW, f"hand{i} wrist")
            if chin is not None:
                cv2.line(bgr, _pt(hand["index"], w, h), _pt(chin, w, h),
                         (180, 180, 0), 1)

        # head pose direction arrow from nose (yaw/pitch -> 2D hint)
        if "yaw" in f:
            nx, ny = _pt((f["face_cx"], f["face_cy"]), w, h)
            dx = int(f["yaw"] * 2.0)
            dy = int(-f["pitch"] * 2.0)
            cv2.arrowedLine(bgr, (nx, ny), (nx + dx, ny + dy), RED, 2, tipLength=0.3)
            cv2.putText(bgr, "head dir", (nx + dx + 5, ny + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    # ---- top-left panel: labels + raw cues ----
    lines = [(f"{DIM_TITLE[d]}: {labels[d][0] if labels[d][0] else '--'}"
              f"  (conf {labels[d][1]:.2f})", GREEN)
             for d in ["attention", "emotion", "action", "posture"]]
    if f.get("face_detected"):
        lines += [
            (f"yaw={f.get('yaw',0):.0f}  pitch={f.get('pitch',0):.0f}  (head_dir arrow)", WHITE),
            (f"smile={f.get('smile',0):.2f}  jawOpen={f.get('jaw_open',0):.2f}  eyeOpen={f.get('eye_open',1):.2f}", WHITE),
            (f"gazeDown={f.get('gaze_down',0):.2f}  gazeSide={f.get('gaze_side',0):.2f}  browUp={f.get('brow_inner_up',0):.2f}", WHITE),
            (f"hand->chin={f.get('hand_to_chin',9):.2f}  hand->face={f.get('hand_to_face',9):.2f}  (x faceHeight)", WHITE),
            (f"hands={f.get('n_hands',0)}", WHITE),
        ]
    else:
        lines += [("FACE NOT DETECTED", RED)]

    pad = 10
    box_h = pad * 2 + 30 * len(lines)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (820, box_h), (0, 0, 0), -1)
    bgr = cv2.addWeighted(overlay, 0.6, bgr, 0.4, 0)
    for i, (t, col) in enumerate(lines):
        cv2.putText(bgr, t, (pad, pad + 22 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2 if col == GREEN else 1)

    cv2.imwrite(out, bgr)
    print("wrote", out)


if __name__ == "__main__":
    import os
    os.makedirs("viz_out", exist_ok=True)
    for t in sys.argv[1:]:
        annotate(f"/tmp/frames/frame_{t}.jpg", f"viz_out/anno_{t}.jpg")
