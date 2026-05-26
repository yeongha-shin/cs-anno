"""
Review web server.

  python review/app.py            # http://127.0.0.1:5000
  python review/app.py --port 5001

Auto-discovers output/<name>.json files and lists them (<name> = video stem).
The reviewer watches the video and edits the 4 labels per segment via
hotkeys/clicks; each change is saved immediately back to the same JSON
(the `auto` original is preserved; `labels`/`reviewed` are updated).
"""
import os, sys, json, glob, argparse
from flask import Flask, jsonify, request, send_file, render_template, abort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from schema import schema_json, SCHEMA, DIMENSIONS, valid_values  # noqa: E402

# Distinct colors cycled per option within a dimension (dashboard bars).
PALETTE = ["#5b9dff", "#4caf50", "#ffb74d", "#e57373", "#ba68c8", "#4dd0e1"]

OUTPUT_DIR = os.path.join(ROOT, "output")
app = Flask(__name__)


def seg_path(name):
    return os.path.join(OUTPUT_DIR, name + ".json")


def load(name):
    p = seg_path(name)
    if not os.path.exists(p):
        abort(404, f"no segments for {name}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save(name, data):
    with open(seg_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_all():
    """(name, data) for every output/<name>.json (skips _-prefixed/unreadable)."""
    out = []
    for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name.startswith("_"):
            continue
        try:
            out.append((name, json.load(open(p, encoding="utf-8"))))
        except Exception:
            continue
    return out


def list_videos(all_data):
    items = []
    for name, d in all_data:
        segs = d.get("segments", [])
        n = d.get("n_segments", len(segs))
        reviewed = sum(1 for s in segs if s.get("reviewed"))
        items.append({"name": name, "video_name": d.get("video_name", name),
                       "n": n, "reviewed": reviewed,
                       "duration": d.get("duration", 0)})
    return items


def compute_stats(all_data):
    """Aggregate label distribution + review progress across all videos."""
    counts = {dim: {v: 0 for v in valid_values(dim)} for dim in DIMENSIONS}
    n_seg = n_reviewed = changed = 0
    for _, d in all_data:
        for s in d.get("segments", []):
            n_seg += 1
            labels = s.get("labels", {})
            for dim in DIMENSIONS:
                if labels.get(dim) in counts[dim]:
                    counts[dim][labels[dim]] += 1
            if s.get("reviewed"):
                n_reviewed += 1
                auto = s.get("auto", {})
                if any(labels.get(dim) != auto.get(dim, {}).get("value")
                       for dim in DIMENSIONS):
                    changed += 1

    dims = []
    for dim in DIMENSIONS:
        total = sum(counts[dim].values())
        opts = []
        for i, (v, label, _desc) in enumerate(SCHEMA[dim]["options"]):
            c = counts[dim][v]
            opts.append({"value": v, "label": label, "count": c,
                         "pct": (100 * c / total) if total else 0,
                         "color": PALETTE[i % len(PALETTE)]})
        dims.append({"key": dim, "title": SCHEMA[dim]["title"],
                     "options": opts, "total": total})

    return {
        "n_videos": len(all_data),
        "n_segments": n_seg,
        "n_reviewed": n_reviewed,
        "progress": (100 * n_reviewed / n_seg) if n_seg else 0,
        "changed": changed,
        "change_pct": (100 * changed / n_reviewed) if n_reviewed else 0,
        "dims": dims,
    }


@app.route("/")
def index():
    all_data = load_all()
    names = [n for n, _ in all_data]
    # ?v=<name> (repeatable) picks which videos feed the stats; default = all.
    picked = [n for n in request.args.getlist("v") if n in names]
    selected = set(picked) if picked else set(names)
    chosen = [(n, d) for n, d in all_data if n in selected]
    return render_template("index.html",
                           videos=list_videos(all_data),
                           stats=compute_stats(chosen),
                           selected=selected)


@app.route("/review/<name>")
def review(name):
    return render_template("review.html", name=name)


@app.route("/api/schema")
def api_schema():
    return jsonify(schema_json())


@app.route("/api/segments/<name>")
def api_segments(name):
    return jsonify(load(name))


@app.route("/api/save/<name>", methods=["POST"])
def api_save(name):
    body = request.get_json(force=True)
    idx = body["idx"]
    data = load(name)
    seg = data["segments"][idx]
    if "labels" in body:
        seg["labels"].update(body["labels"])
    if "reviewed" in body:
        seg["reviewed"] = bool(body["reviewed"])
    save(name, data)
    reviewed = sum(1 for s in data["segments"] if s.get("reviewed"))
    return jsonify({"ok": True, "reviewed": reviewed, "n": len(data["segments"])})


@app.route("/video/<name>")
def video(name):
    d = load(name)
    # Portable lookup: prefer a local data/<video_name> (works after clone on any
    # machine), then fall back to the absolute path recorded at extraction time.
    vn = d.get("video_name")
    candidates = []
    if vn:
        candidates.append(os.path.join(ROOT, "data", vn))
    candidates.append(d.get("video", ""))
    for path in candidates:
        if path and os.path.exists(path):
            return send_file(path, conditional=True)   # Range requests -> seeking works
    abort(404, f"video not found. Put '{vn}' into the data/ folder.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"  review server: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
