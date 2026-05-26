"""
Per-frame visual feature extraction.

Runs three MediaPipe Tasks models on a single frame:
  - FaceLandmarker : 478 landmarks + 52 blendshapes + head-pose matrix
  - HandLandmarker : up to 2 hands x 21 landmarks + handedness
  - PoseLandmarker : 33 upper-body landmarks

Output is a flat feature dict consumed directly by labelers.py.
All coordinates are normalized 0..1 (image fraction), so distances between
face/hand/pose points are directly comparable.
"""
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- face mesh landmark indices ---
CHIN = 152          # chin tip
FOREHEAD = 10       # top of forehead
NOSE_TIP = 1
L_CHEEK = 454
R_CHEEK = 234
MOUTH_L = 61
MOUTH_R = 291

# --- hand landmark indices ---
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12

# --- pose landmark indices ---
P_NOSE = 0
P_L_SH = 11
P_R_SH = 12


def _euler_from_matrix(M):
    """4x4 facial transformation matrix -> (pitch, yaw, roll) in degrees.

    pitch: head up(+)/down(-), yaw: left/right turn, roll: tilt.
    Sign consistency matters more than absolute interpretation.
    """
    R = np.array(M)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    else:
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
        roll = math.atan2(-R[1, 2], R[1, 1])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class FeatureExtractor:
    def __init__(self, model_dir="models"):
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=f"{model_dir}/face_landmarker.task"),
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=3))   # detect several, then keep the largest (= target child)
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=f"{model_dir}/hand_landmarker.task"),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2))
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=f"{model_dir}/pose_landmarker.task"),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1))

    def close(self):
        for m in (self.face, self.hands, self.pose):
            try:
                m.close()
            except Exception:
                pass

    def extract(self, rgb):
        """rgb: HxWx3 uint8 (RGB). -> feature dict."""
        mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        f = {"face_detected": False, "n_hands": 0}

        # ---------- face ----------
        fr = self.face.detect(mpimg)
        face_pts = None
        if fr.face_landmarks:
            # of all detected faces, keep the largest (forehead-to-chin height) = target child
            def _fh(lmk):
                return abs(lmk[FOREHEAD].y - lmk[CHIN].y)
            ti = max(range(len(fr.face_landmarks)),
                     key=lambda i: _fh(fr.face_landmarks[i]))
            f["face_detected"] = True
            f["n_faces"] = len(fr.face_landmarks)
            lm = fr.face_landmarks[ti]
            face_pts = [(p.x, p.y) for p in lm]
            chin = face_pts[CHIN]
            forehead = face_pts[FOREHEAD]
            face_h = max(_dist(chin, forehead), 1e-4)   # face height = normalized distance unit
            f["face_size"] = face_h
            f["face_cx"] = (forehead[0] + chin[0]) / 2
            f["face_cy"] = (forehead[1] + chin[1]) / 2
            f["chin"] = chin
            xs = [p[0] for p in face_pts]
            ys = [p[1] for p in face_pts]
            f["face_bbox"] = (min(xs), min(ys), max(xs), max(ys))  # face region

            if fr.facial_transformation_matrixes:
                pitch, yaw, roll = _euler_from_matrix(
                    fr.facial_transformation_matrixes[ti])
                f["pitch"], f["yaw"], f["roll"] = pitch, yaw, roll

            if fr.face_blendshapes:
                bs = {b.category_name: b.score for b in fr.face_blendshapes[ti]}
                f["bs"] = bs
                g = bs.get
                f["smile"] = (g("mouthSmileLeft", 0) + g("mouthSmileRight", 0)) / 2
                f["jaw_open"] = g("jawOpen", 0)
                f["eye_blink"] = (g("eyeBlinkLeft", 0) + g("eyeBlinkRight", 0)) / 2
                f["eye_open"] = 1.0 - f["eye_blink"]
                f["gaze_down"] = (g("eyeLookDownLeft", 0) + g("eyeLookDownRight", 0)) / 2
                f["gaze_up"] = (g("eyeLookUpLeft", 0) + g("eyeLookUpRight", 0)) / 2
                f["gaze_side"] = (g("eyeLookOutLeft", 0) + g("eyeLookOutRight", 0)
                                   + g("eyeLookInLeft", 0) + g("eyeLookInRight", 0)) / 2
                f["brow_down"] = (g("browDownLeft", 0) + g("browDownRight", 0)) / 2
                f["brow_inner_up"] = g("browInnerUp", 0)
                f["mouth_frown"] = (g("mouthFrownLeft", 0) + g("mouthFrownRight", 0)) / 2
                f["mouth_press"] = (g("mouthPressLeft", 0) + g("mouthPressRight", 0)) / 2

        # ---------- hands ----------
        hr = self.hands.detect(mpimg)
        hands = []
        if hr.hand_landmarks:
            for pts in hr.hand_landmarks:
                hands.append({
                    "wrist": (pts[WRIST].x, pts[WRIST].y),
                    "index": (pts[INDEX_TIP].x, pts[INDEX_TIP].y),
                    "middle": (pts[MIDDLE_TIP].x, pts[MIDDLE_TIP].y),
                    "thumb": (pts[THUMB_TIP].x, pts[THUMB_TIP].y),
                })
        f["n_hands"] = len(hands)
        f["hands"] = hands

        # hand-to-face / hand-to-chin proximity (normalized by face size)
        if face_pts and hands:
            face_h = f["face_size"]
            chin = f["chin"]
            cx, cy = f["face_cx"], f["face_cy"]
            min_chin = min_face = 1e9
            for h in hands:
                for key in ("index", "middle", "wrist", "thumb"):
                    min_chin = min(min_chin, _dist(h[key], chin) / face_h)
                    min_face = min(min_face, _dist(h[key], (cx, cy)) / face_h)
            f["hand_to_chin"] = min_chin
            f["hand_to_face"] = min_face

        # ---------- pose (upper body) ----------
        pr = self.pose.detect(mpimg)
        if pr.pose_landmarks:
            pp = pr.pose_landmarks[0]
            ls, rs = pp[P_L_SH], pp[P_R_SH]
            f["shoulder_width"] = abs(ls.x - rs.x)
            f["shoulder_tilt"] = ls.y - rs.y
            f["shoulder_cy"] = (ls.y + rs.y) / 2
            f["nose_y"] = pp[P_NOSE].y
            f["nose_below_shoulder"] = pp[P_NOSE].y - f["shoulder_cy"]
        return f
