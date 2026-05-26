# Reviewer Guide

Your role: review the auto-generated labels on learning videos and correct them.
You do **not** need the model / extraction setup — only a lightweight web server.

---

## 1. Prerequisites
- **Python 3.8+** — check with `python3 --version`
- **pip** (comes with Python)

That's it. No GPU, no MediaPipe, no OpenCV.

---

## 2. Get the code
```bash
git clone https://github.com/yeongha-shin/cs-anno.git
cd cs-anno
```
The label files (`output/*.json`) are included in the repo.

---

## 3. Install (Flask only)
```bash
pip install flask
```
> MediaPipe / OpenCV are **not** needed for review. They are only used to
> generate labels, which is already done.

---

## 4. Download the video files into `data/`
Video files are large and are **not** in the repo. Download them from here:

**➡️ Videos (Google Drive): https://drive.google.com/drive/folders/1YvVIonG4u3Z7uphMmCAO4wa7g5NX2i3z?usp=sharing**

- Download every video and put it into the **`data/`** folder.
- **Keep the exact filenames** as provided (e.g. `kid_2.mov`, `kid_3.mp4`,
  `adult_4.mp4`). Do not rename them — the filename must match what each label
  file expects, or the video will not load.

Expected layout after download:
```
data/kid_2.mov
data/kid_3.mp4
data/adult_4.mp4
```

---

## 5. Run the review server
```bash
python review/app.py
```
Then open **http://127.0.0.1:5000** in your browser (Chrome recommended).

- Different port if 5000 is taken: `python review/app.py --port 5001`

---

## 6. How to review
1. Pick a video from the list.
2. The video **loops over the current 5-second segment** automatically.
3. Each of the 4 labels shows the **auto suggestion** (marked with an orange dot):
   - **Attention** — 1 Low / 2 Medium / 3 High
   - **Emotion** — Happy / Boring / Frustrated / Thinking
   - **Action** — Watching / Touching / Speaking
   - **Posture** — Upright / Chin rest / Face touching
4. If a label is wrong, **click the correct option** or press its shortcut key.
   (Hover an option to see its definition.)
5. Move on. **Your edits save automatically and instantly** to `output/<name>.json`.

### Shortcuts
| Action | Key |
|---|---|
| Attention | `1` `2` `3` |
| Emotion | `q` happy · `w` boring · `e` frustrated · `r` thinking |
| Action | `a` watching · `s` touching · `d` speaking |
| Posture | `z` upright · `x` chin_rest · `c` face_touching |
| Prev / Next segment | `←` / `→` |
| Play / Pause | `Space` |
| **Mark reviewed + next** | `Enter` |

### Important: editing vs. marking reviewed
- **Editing a label** → saved immediately (even if you just move on with `←`/`→`).
- **`Enter`** → also marks the segment as **reviewed** (the progress counter goes up).
- Use the **"order: low confidence first"** dropdown to review the most uncertain
  segments first.

Orange dot = the auto suggestion. `✎ edited` = you changed it from the auto value.

---

## 7. Send your results back
All your work is saved in `output/<name>.json` (the `labels` field, plus a
`reviewed` flag per segment). To return it:
- **Git:** `git add output && git commit -m "review <name>" && git push`
  (you need write access or a Personal Access Token), **or**
- Send the `output/*.json` files back to the project owner directly.

---

## Troubleshooting
- **Video won't play / "video not found"** — the file isn't in `data/`, or the
  filename doesn't match. See step 4.
- **Port already in use** — `python review/app.py --port 5001`.
- **Label meanings** — hover an option for its definition; the full codebook is in
  `schema.py`.
