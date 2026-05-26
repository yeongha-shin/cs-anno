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
from schema import schema_json  # noqa: E402

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


def list_videos():
    items = []
    for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name.startswith("_"):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        n = d.get("n_segments", len(d.get("segments", [])))
        reviewed = sum(1 for s in d["segments"] if s.get("reviewed"))
        items.append({"name": name, "video_name": d.get("video_name", name),
                       "n": n, "reviewed": reviewed,
                       "duration": d.get("duration", 0)})
    return items


@app.route("/")
def index():
    return render_template("index.html", videos=list_videos())


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
