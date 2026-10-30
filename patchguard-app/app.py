"""
PatchGuard web app - interactive front-end for the PatchGuard defence.

Run:  python app.py        ->  http://127.0.0.1:5000

The defence itself (patch_mask / neutralise) is imported unchanged from
patchguard_demo.py. Person detection with YOLOv8n is optional: if `ultralytics`
is installed the app also shows raw vs. sanitised detections and the
discrepancy alert (|D_def| > |D_raw|); otherwise it runs defence-only.
"""
import base64, io, json, os, time

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from patchguard_demo import patch_mask, neutralise

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SIDE = 1024
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ---------------------------------------------------------------- helpers
def to_b64(arr, fmt="PNG"):
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    return "data:image/%s;base64,%s" % (fmt.lower(), base64.b64encode(buf.getvalue()).decode())


def load_image(file_storage):
    im = Image.open(file_storage.stream).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.array(im)


def density_map(img_u8, win):
    """Local gradient density rho (same maths as patch_mask) for the heat-map view."""
    g = img_u8.astype(np.float32).mean(axis=2) / 255.0
    mag = np.hypot(ndi.sobel(g, axis=1), ndi.sobel(g, axis=0))
    return ndi.uniform_filter(mag, size=win)


def heatmap(rho, k):
    thr = rho.mean() + k * rho.std()
    v = np.clip(rho / (rho.max() + 1e-9), 0, 1)
    # simple black -> orange -> yellow ramp
    r = np.clip(v * 2.0, 0, 1); g = np.clip(v * 2.0 - 0.6, 0, 1); b = np.clip(v * 2.0 - 1.4, 0, 1) * 0.6
    out = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    return out, float(thr)


def overlay_mask(img, m):
    out = img.astype(np.float32).copy()
    out[m] = out[m] * 0.35 + np.array([255, 60, 60], np.float32) * 0.65
    return out.astype(np.uint8)


def add_sticker(img, cx, cy, size, seed=7):
    """Synthetic high-frequency sticker (stand-in for a learned adversarial patch)."""
    h, w = img.shape[:2]
    side = max(24, int(size * min(h, w)))
    custom = os.path.join(HERE, "results", "patch.png")
    if os.path.exists(custom):                      # use the real learned patch if present
        p = np.array(Image.open(custom).convert("RGB").resize((side, side), Image.BILINEAR))
    else:
        rng = np.random.default_rng(seed)
        cells = rng.integers(0, 256, (max(8, side // 8), max(8, side // 8), 3), dtype=np.uint8)
        p = np.array(Image.fromarray(cells).resize((side, side), Image.NEAREST))
        p = np.clip(p.astype(np.int16) + rng.integers(-40, 40, p.shape), 0, 255).astype(np.uint8)
    x1 = int(np.clip(cx * w - side / 2, 0, w - side)); y1 = int(np.clip(cy * h - side / 2, 0, h - side))
    out = img.copy(); out[y1:y1 + side, x1:x1 + side] = p
    return out


def sample_scene():
    """Procedural street-ish scene so the app can be demoed without any upload."""
    h, w = 480, 640
    y = np.linspace(0, 1, h)[:, None, None]
    sky = np.array([170, 200, 230]) * (1 - y) + np.array([210, 205, 195]) * y
    img = np.broadcast_to(sky, (h, w, 3)).copy()
    img[300:] = np.array([95, 98, 105])
    rng = np.random.default_rng(1)
    for x0 in range(20, 620, 110):
        bh = int(rng.integers(90, 220)); img[300 - bh:300, x0:x0 + 90] = rng.integers(110, 170, 3)
    im = Image.fromarray(img.astype(np.uint8)); d = ImageDraw.Draw(im)
    for cx in (210, 430):                                     # two stylised pedestrians
        d.ellipse([cx - 16, 210, cx + 16, 244], fill=(224, 180, 150))
        d.rounded_rectangle([cx - 24, 244, cx + 24, 340], 10, fill=(60, 80, 140))
        d.rectangle([cx - 20, 340, cx - 4, 420], fill=(40, 40, 50)); d.rectangle([cx + 4, 340, cx + 20, 420], fill=(40, 40, 50))
    return np.array(im)


# ------------------------------------------------------- optional detector
_yolo = {"model": None, "tried": False, "err": None}

def get_yolo():
    if not _yolo["tried"]:
        _yolo["tried"] = True
        try:
            from ultralytics import YOLO
            _yolo["model"] = YOLO(os.environ.get("YOLO_WEIGHTS", "yolov8n.pt"))
        except Exception as e:                       # not installed / no weights / offline
            _yolo["err"] = f"{type(e).__name__}: {e}"
    return _yolo["model"]


def detect(img_u8, conf):
    model = get_yolo()
    r = model.predict(img_u8[..., ::-1], conf=conf, classes=[0], verbose=False)[0]
    return [list(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()], [float(c) for c in r.boxes.conf.cpu().numpy()]


def draw_boxes(img, boxes, confs, color):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        d.text((x1 + 4, y1 + 2), f"person {c:.2f}", fill=color)
    return np.array(im)


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    ok = get_yolo() is not None
    return jsonify(detector=ok, detector_error=None if ok else _yolo["err"],
                   learned_patch=os.path.exists(os.path.join(HERE, "results", "patch.png")))


@app.get("/api/results")
def results():
    p = os.path.join(HERE, "results", "results.json")
    if os.path.exists(p):
        return jsonify(source="results.json", data=json.load(open(p)))
    # figures reported in the project README / report
    f = lambda p_, r_, f_: dict(precision=p_, recall=r_, f1=f_)
    return jsonify(source="README", data={
        "clean": f(.986, 1.0, .993), "random": f(.978, .922, .949), "attacked": f(.85, .239, .374),
        "def_clean": f(.972, .965, .968), "def_attacked": f(.945, .845, .892),
        "attack_success_rate": .761, "attack_success_rate_after_defence": .155,
        "alert_rate_attacked": .8235, "false_alert_rate_clean": .0441,
        "latency_ms_detector_only": 18.42, "latency_ms_patchguard_pipeline": 33.24,
        "test_images_used": 68})


@app.post("/api/analyze")
def analyze():
    try:
        if request.files.get("image"):
            img = load_image(request.files["image"])
        else:
            img = sample_scene()
        f = request.form
        win = int(f.get("win", 31)) | 1                       # keep odd
        k = float(f.get("k", 2.0)); dilate = int(f.get("dilate", 9))
        min_area = int(f.get("min_area", 400)); sigma = float(f.get("sigma", 12))
        conf = float(f.get("conf", 0.5))
        original = img
        if f.get("sticker") == "1":
            img = add_sticker(img, float(f.get("sx", .5)), float(f.get("sy", .5)), float(f.get("ss", .2)))

        t0 = time.perf_counter()
        mask = patch_mask(img, win=win, k=k, dilate=dilate, min_area=min_area)
        clean = neutralise(img, mask, sigma=sigma)
        t_def = (time.perf_counter() - t0) * 1000

        rho = density_map(img, win)
        heat, thr = heatmap(rho, k)
        out = {
            "size": [img.shape[1], img.shape[0]],
            "images": {"original": to_b64(original), "input": to_b64(img), "density": to_b64(heat),
                       "mask": to_b64((mask * 255).astype(np.uint8)), "overlay": to_b64(overlay_mask(img, mask)),
                       "sanitised": to_b64(clean)},
            "metrics": {"mask_coverage_pct": float(mask.mean() * 100), "defence_ms": t_def,
                        "threshold": thr, "rho_mean": float(rho.mean()), "rho_max": float(rho.max())},
            "detection": None,
        }
        if f.get("detect") == "1":
            if get_yolo() is None:
                out["detection"] = {"available": False, "error": _yolo["err"]}
            else:
                t0 = time.perf_counter(); b_raw, c_raw = detect(img, conf); t_raw = (time.perf_counter() - t0) * 1000
                b_def, c_def = detect(clean, conf)
                out["detection"] = {
                    "available": True, "raw_count": len(b_raw), "def_count": len(b_def),
                    "alert": len(b_def) > len(b_raw), "detector_ms": t_raw,
                    "raw_img": to_b64(draw_boxes(img, b_raw, c_raw, (230, 60, 60))),
                    "def_img": to_b64(draw_boxes(clean, b_def, c_def, (60, 200, 90)))}
        return jsonify(out)
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
