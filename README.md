# Child learning-video annotation (auto pre-labeling + human review)

One child per video (the center / largest face) is labeled across 4 dimensions.
MediaPipe produces auto pre-labels, then a human reviews/edits them per segment
in a web UI.

## Label dimensions (definitions in `schema.py`)
| dimension | options |
|---|---|
| attention | `1_low` / `2_medium` / `3_high` |
| emotion | `happy` / `boring` / `frustrated` / `thinking` (neutral default) |
| action | `watching` / `touching` (hand on the tablet area) / `speaking` (mouth moving) |
| posture | `upright` / `chin_rest` (hand supports chin) / `face_touching` (hand on face) |

## Install
```bash
pip install mediapipe opencv-contrib-python flask
```

### Download models
The MediaPipe Tasks bundles are not committed. Download them into `models/`:
```bash
mkdir -p models && cd models
base="https://storage.googleapis.com/mediapipe-models"
curl -sSL -o face_landmarker.task "$base/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
curl -sSL -o hand_landmarker.task "$base/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
curl -sSL -o pose_landmarker.task "$base/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
```

## 1) Extract auto pre-labels
```bash
python extract.py data/kid_2.mov            # -> output/kid_2.json
python extract.py data/xxx.mov --seg 5.0 --fps 5
```
- `--seg` segment length in seconds (default 5.0), `--fps` analysis sampling fps (default 5)
- ~4 min per 10-minute video (on GPU)

## 2) Review UI
```bash
python review/app.py            # http://127.0.0.1:5000
```
- Pick a video from the list -> auto labels are shown per segment
- Shortcuts: `←/→` move segment, `Space` play/pause, `Enter` mark reviewed + next.
  Set each label with the key shown on its option (attention 1/2/3, emotion q/w/e/r, etc.)
- Orange dot = auto suggestion, `✎ edited` = changed from auto
- Set order to "low confidence first" to review uncertain segments first (active learning)
- Every change is saved immediately to `output/<name>.json` (the `auto` original is kept)

## Tuning the auto labels
- Thresholds: top of `pipeline/labelers.py` (e.g. `SMILE_HAPPY`, `EYE_CLOSED`,
  `CHIN_REST_DIST`, `IPAD_Y`, `FACE_BOX_MARGIN`, `SPEAK_JAW_STD`)
- Inspect per-frame behavior: `python viz_check.py 30 90 420` -> `viz_out/anno_*.jpg`

## Layout
```
schema.py            label codebook (shared by pipeline + UI)
extract.py           video -> segment JSON
pipeline/features.py MediaPipe feature extraction (face/hands/pose)
pipeline/labelers.py features -> label + confidence rules
review/app.py        review Flask server
viz_check.py         per-frame visual inspection tool
output/<name>.json   auto labels + review results
```

## Output JSON (review results)
Each segment has: `auto` (frozen original), `labels` (editable, human-corrected),
`reviewed` (done flag). For training data, use `labels` of segments where
`reviewed: true`.
