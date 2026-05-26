"""
Map a feature dict -> label + confidence for the 4 dimensions (heuristics).

Design
  - All thresholds are module-level constants so they are easy to tune
    after looking at review results.
  - Each labeler returns (value, confidence), confidence in 0..1 expressing
    "how clearly the cue crossed the threshold + was the signal present".
  - Weak dimensions get lower confidence so the reviewer sees them first
    (low-confidence-first review).
This is a *starting point*, not ground truth -- the human corrects it.

Note: `action` "speaking" is decided per-segment (mouth movement over the
window), so it lives in aggregate_action(), not the per-frame labeler.
"""
import math
import statistics
from schema import valid_values

# ---- posture / action geometry (hand vs face / tablet) ----
CHIN_REST_DIST = 0.5      # hand-to-chin (in face-heights); also needs wrist below chin
FACE_BOX_MARGIN = 0.10    # expand face bbox by this fraction; hand inside => face_touching
IPAD_Y = 0.80             # if a hand's lowest point y exceeds this (lower screen) => touching
YAW_AWAY = 28.0           # |yaw| above this = head turned from screen (attention penalty, deg)

# ---- emotion ----
SMILE_HAPPY = 0.20        # mouth-smile blendshape above this => happy
EYE_CLOSED = 0.45         # eyeOpen below this (eyes nearly closed) => boring
BROW_CONFUSED = 0.45      # lowered brows above this => frustrated cue
FROWN_FRUSTRATED = 0.30   # mouth frown above this => frustrated

# ---- action (segment-level) ----
SPEAK_JAW_STD = 0.06      # std of jawOpen across a segment above this => mouth moving
SPEAK_JAW_RANGE = 0.14    # and max-min of jawOpen above this => speaking


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def label_posture(f):
    # options: upright / chin_rest / face_touching
    if not f.get("face_detected"):
        return (None, 0.0)                       # undeterminable -> excluded from vote

    hands = f.get("hands", [])
    chin = f.get("chin")
    face_h = f.get("face_size", 1.0) or 1.0

    # chin_rest (strict): the hand must support the chin from below
    #   (1) hand close to chin, (2) wrist below chin (arm comes from below),
    #   (3) the closest hand point is at/below chin (the contact point).
    #   -> finger in mouth (contact above chin) and hand behind head (wrist above chin) excluded.
    if chin and hands:
        cy = chin[1]
        for hand in hands:
            pts = [hand[k] for k in ("index", "middle", "thumb", "wrist")]
            closest = min(pts, key=lambda p: math.hypot(p[0] - chin[0], p[1] - chin[1]))
            d = math.hypot(closest[0] - chin[0], closest[1] - chin[1]) / face_h
            wrist_below_chin = hand["wrist"][1] > cy
            contact_below_chin = closest[1] >= cy - 0.05     # allow up to 0.05 above chin
            if d < CHIN_REST_DIST and wrist_below_chin and contact_below_chin:
                return ("chin_rest", _clamp(0.55 + (CHIN_REST_DIST - d)))

    # face_touching: a hand point must actually fall inside the face region (bbox),
    #   not just be close to the center -> a hand beside the face is excluded.
    bbox = f.get("face_bbox")
    if bbox and hands:
        x0, y0, x1, y1 = bbox
        mx = (x1 - x0) * FACE_BOX_MARGIN          # margin for edge contact (cheek/jawline)
        my = (y1 - y0) * FACE_BOX_MARGIN
        for hand in hands:
            for k in ("index", "middle", "thumb", "wrist"):
                px, py = hand[k]
                if x0 - mx <= px <= x1 + mx and y0 - my <= py <= y1 + my:
                    return ("face_touching", 0.55)
    return ("upright", 0.6)


def label_attention(f):
    if not f.get("face_detected"):
        return ("1_low", 0.5)

    yaw = abs(f.get("yaw", 0.0))
    eye_open = f.get("eye_open", 1.0)
    gaze_down = f.get("gaze_down", 0.0)
    gaze_side = f.get("gaze_side", 0.0)

    score = 0.5
    # facing front + looking down (typical tablet-viewing pose) -> bonus
    if yaw < 18 and (gaze_down > 0.08 or f.get("pitch", 0) < 5):
        score += 0.30
    if yaw > YAW_AWAY:
        score -= 0.40
    if gaze_side > 0.55:
        score -= 0.20
    if eye_open < 0.40:           # eyes closed / nearly closed
        score -= 0.35
    # hand in the writing/task area (low) while facing the screen -> engagement bonus
    if f.get("hand_to_face", 9) > 1.3 and f.get("n_hands", 0) > 0 and yaw < 20:
        score += 0.10
    score = _clamp(score)

    if score < 0.40:
        value, ref = "1_low", 0.40
    elif score < 0.70:
        value, ref = "2_medium", 0.55
    else:
        value, ref = "3_high", 0.70
    conf = _clamp(0.4 + abs(score - ref) * 1.5)
    return (value, conf)


def label_emotion(f):
    # options: happy / boring / frustrated / thinking (thinking = neutral default)
    if not f.get("face_detected"):
        return (None, 0.0)                       # undeterminable -> excluded from vote

    smile = f.get("smile", 0.0)
    brow_down = f.get("brow_down", 0.0)
    frown = f.get("mouth_frown", 0.0)
    press = f.get("mouth_press", 0.0)
    eye_open = f.get("eye_open", 1.0)

    # happy: lips form a smile (highest priority)
    if smile > SMILE_HAPPY:
        return ("happy", _clamp(0.4 + smile))
    # frustrated: pressed lips / frown / lowered brows
    if frown > FROWN_FRUSTRATED or press > 0.5 or brow_down > BROW_CONFUSED:
        return ("frustrated", _clamp(0.35 + max(frown, press, brow_down)))
    # boring: only when eyes are nearly closed (drowsy / slumping)
    if eye_open < EYE_CLOSED:
        return ("boring", _clamp(0.4 + (EYE_CLOSED - eye_open)))
    # otherwise neutral (eyes open, no smile/frown) -> thinking
    return ("thinking", 0.4)


def label_action(f):
    # per-frame options: watching / touching   (speaking is segment-level)
    # touching = a hand reaches the lower screen area (tablet).
    if not f.get("face_detected"):
        return (None, 0.0)                       # undeterminable -> excluded from vote

    lowest_hand_y = 0.0
    for hand in f.get("hands", []):
        for key in ("wrist", "index", "middle", "thumb"):
            lowest_hand_y = max(lowest_hand_y, hand[key][1])
    if lowest_hand_y > IPAD_Y:
        return ("touching", _clamp(0.45 + (lowest_hand_y - IPAD_Y) * 1.5))
    return ("watching", 0.45)


LABELERS = {
    "attention": label_attention,
    "emotion": label_emotion,
    "action": label_action,
    "posture": label_posture,
}


def aggregate(frame_values):
    """List of per-frame [(value, conf), ...] -> (representative value, segment confidence).

    Representative = confidence-weighted mode. Segment confidence =
    (agreement ratio) x (mean frame confidence); jittery segments score lower.
    """
    frame_values = [(v, c) for v, c in frame_values if v is not None]
    if not frame_values:
        return (None, 0.0)               # whole segment undeterminable -> unlabeled
    weight = {}
    for v, c in frame_values:
        weight[v] = weight.get(v, 0.0) + c
    best = max(weight, key=weight.get)
    agree = sum(c for v, c in frame_values if v == best)
    total = sum(c for v, c in frame_values) or 1e-9
    mean_conf = sum(c for v, c in frame_values) / len(frame_values)
    seg_conf = (agree / total) * mean_conf
    return (best, round(seg_conf, 3))


def aggregate_action(frame_values, jaw_values):
    """Segment-level action: detect speaking from mouth movement over the window.

    jaw_values: jawOpen per analyzed (face-present) frame in the segment.
    If the mouth opens/closes enough across the segment -> speaking, unless a
    hand is clearly on the tablet the whole time (touching wins).
    """
    base_val, base_conf = aggregate(frame_values)
    if len(jaw_values) >= 3:
        std = statistics.pstdev(jaw_values)
        rng = max(jaw_values) - min(jaw_values)
        if std > SPEAK_JAW_STD and rng > SPEAK_JAW_RANGE:
            if base_val == "touching" and base_conf > 0.5:
                return (base_val, base_conf)     # hand firmly on tablet -> keep touching
            return ("speaking", round(_clamp(0.45 + std), 3))
    return (base_val, base_conf)
