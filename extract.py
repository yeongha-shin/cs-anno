"""
Video -> per-segment auto-label JSON.

Splits the video into segment_seconds windows (default 5s), samples frames at
sample_fps within each window, extracts features -> per-frame labels -> aggregates
to a segment label.

Output JSON (consumed directly by the review UI):
  segments[i] = {
    idx, start, end, n_frames, face_rate,
    auto   : {dim: {value, confidence}},   # auto labels (original, frozen)
    labels : {dim: value},                 # for review (= copy of auto, human edits)
    reviewed: false
  }

Usage:
  python extract.py data/kid_2.mov                       # -> output/kid_2.json
  python extract.py data/kid_2.mov --seg 5.0 --fps 5 --out output/kid_2.json
"""
import os, sys, json, time, argparse
import cv2

from pipeline.features import FeatureExtractor
from pipeline.labelers import (LABELERS, aggregate, aggregate_action,
                               aggregate_attention, aggregate_emotion)
from schema import DIMENSIONS, default_value


def extract_video(video_path, out_path, segment_seconds=5.0, sample_fps=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    step = max(1, round(fps / sample_fps))          # analyze every Nth frame
    n_segments = int(duration // segment_seconds) + (1 if duration % segment_seconds else 0)

    print(f"[extract] {video_path}")
    print(f"  fps={fps:.1f} frames={total} duration={duration:.1f}s "
          f"seg={segment_seconds}s sample_fps={sample_fps} -> {n_segments} segments")

    ex = FeatureExtractor()
    # seg_idx -> {dim: [(value, conf), ...]} + jawOpen list + face/frame counts
    buckets = [{d: [] for d in DIMENSIONS} for _ in range(n_segments)]
    jaw = [[] for _ in range(n_segments)]
    motion = [[] for _ in range(n_segments)]    # (face_cx, face_cy) per frame -> body movement
    face_hit = [0] * n_segments
    frame_cnt = [0] * n_segments

    fidx = 0
    t0 = time.time()
    while True:
        if not cap.grab():
            break
        if fidx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                seg = int((fidx / fps) // segment_seconds)
                if 0 <= seg < n_segments:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    f = ex.extract(rgb)
                    frame_cnt[seg] += 1
                    if f.get("face_detected"):
                        face_hit[seg] += 1
                        jaw[seg].append(f.get("jaw_open", 0.0))
                        if "face_cx" in f:
                            motion[seg].append((f["face_cx"], f["face_cy"]))
                    for dim, fn in LABELERS.items():
                        buckets[seg][dim].append(fn(f))
                    if frame_cnt[seg] == 1 and seg % 10 == 0:
                        print(f"  seg {seg}/{n_segments}  ({time.time()-t0:.0f}s)", flush=True)
        fidx += 1
    cap.release()
    ex.close()

    segments = []
    for i in range(n_segments):
        auto, labels = {}, {}
        for dim in DIMENSIONS:
            if dim == "action":
                value, conf = aggregate_action(buckets[i][dim], jaw[i])
            elif dim == "attention":
                value, conf = aggregate_attention(buckets[i][dim], motion[i])
            elif dim == "emotion":
                value, conf = aggregate_emotion(buckets[i][dim], motion[i])
            else:
                value, conf = aggregate(buckets[i][dim])
            auto[dim] = {"value": value, "confidence": conf}
            labels[dim] = value if value is not None else default_value(dim)
        segments.append({
            "idx": i,
            "start": round(i * segment_seconds, 3),
            "end": round(min((i + 1) * segment_seconds, duration), 3),
            "n_frames": frame_cnt[i],
            "face_rate": round(face_hit[i] / frame_cnt[i], 2) if frame_cnt[i] else 0.0,
            "auto": auto,
            "labels": labels,
            "reviewed": False,
        })

    out = {
        "video": os.path.abspath(video_path),
        "video_name": os.path.basename(video_path),
        "fps": round(fps, 3),
        "duration": round(duration, 3),
        "segment_seconds": segment_seconds,
        "sample_fps": sample_fps,
        "n_segments": n_segments,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "segments": segments,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"[done] {time.time()-t0:.0f}s -> {out_path}  ({n_segments} segments)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seg", type=float, default=5.0, help="segment length (seconds)")
    ap.add_argument("--fps", type=float, default=5.0, help="analysis sampling fps")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()
    out_path = args.out or f"output/{os.path.splitext(os.path.basename(args.video))[0]}.json"
    extract_video(args.video, out_path, args.seg, args.fps)


if __name__ == "__main__":
    main()
