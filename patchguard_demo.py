"""
PatchGuard micro-project demo (Trustworthy & Responsible AI, SAKEC)
Author : Swayam Netke | Roll 33 | PRN 124BTAI1066
Pipeline: (1) learn a universal adversarial sticker against YOLOv8n (person class)
          (2) defend with gradient-density patch localisation + neutralisation
          (3) report recall / attack success rate / false alerts / latency

Usage (Colab GPU recommended):
    pip install ultralytics scipy matplotlib
    python patchguard_demo.py --images ./PennFudanPed/PNGImages --epochs 30
"""
import argparse, glob, json, os, random, time
import numpy as np
from scipy import ndimage as ndi

# --------------------------------------------------------------------------
# DEFENCE (pure NumPy/SciPy - no deep-learning dependency)
# --------------------------------------------------------------------------
def patch_mask(img_u8, win=31, k=2.0, dilate=9, min_area=400):
    """Locate regions with abnormally dense high-frequency gradients (typical of printed patches)."""
    g = img_u8.astype(np.float32).mean(axis=2) / 255.0
    gx, gy = ndi.sobel(g, axis=1), ndi.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    rho = ndi.uniform_filter(mag, size=win)             # local gradient density
    m = rho > (rho.mean() + k * rho.std())               # adaptive threshold
    lab, n = ndi.label(m)                                # drop tiny blobs
    if n:
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        keep = np.isin(lab, 1 + np.where(sizes >= min_area)[0])
        m = keep
    return ndi.binary_dilation(m, iterations=dilate)

def neutralise(img_u8, m, sigma=12):
    """Replace masked pixels by a normalised-convolution blur of the surrounding clean pixels."""
    if not m.any():
        return img_u8
    f = img_u8.astype(np.float32)
    w = (~m).astype(np.float32)[..., None]
    num = ndi.gaussian_filter(f * w, sigma=(sigma, sigma, 0))
    den = ndi.gaussian_filter(w, sigma=(sigma, sigma, 0)) + 1e-6
    fill = num / den
    out = np.where(m[..., None], fill, f)
    return np.clip(out, 0, 255).astype(np.uint8)

def patchguard(img_u8, **kw):
    m = patch_mask(img_u8, **kw)
    return neutralise(img_u8, m), m

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def recall(ref, det, thr=0.5):
    if not ref: return None
    return sum(any(iou(r, d) >= thr for d in det) for r in ref) / len(ref)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--split", type=float, default=0.6, help="fraction used to train the patch")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--win", type=int, default=31)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch, torch.nn.functional as F
    from PIL import Image
    from ultralytics import YOLO
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(a.weights)
    net = yolo.model.to(dev).float().eval()
    for p in net.parameters(): p.requires_grad_(False)

    S = 640
    paths = sorted(glob.glob(os.path.join(a.images, "*.png")) + glob.glob(os.path.join(a.images, "*.jpg")))
    random.shuffle(paths)
    imgs = [np.array(Image.open(p).convert("RGB").resize((S, S))) for p in paths]
    n_tr = int(len(imgs) * a.split)
    train, test = imgs[:n_tr], imgs[n_tr:]
    print(f"images={len(imgs)} train={len(train)} test={len(test)} device={dev}")

    def detect(img_u8):
        r = yolo.predict(img_u8[..., ::-1], conf=a.conf, classes=[0], imgsz=S, verbose=False, device=dev)[0]
        return [list(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()]

    def to_t(img_u8): return torch.from_numpy(img_u8).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)

    def place(x, boxes, patch, eot=False):
        x = x.clone()
        for (x1, y1, x2, y2) in boxes:
            w = x2 - x1; side = int(max(16, 0.7 * w))
            if eot: side = int(side * random.uniform(0.9, 1.1))
            p = F.interpolate(patch, size=(side, side), mode="bilinear", align_corners=False)
            if eot:
                p = p * random.uniform(0.8, 1.2) + 0.03 * torch.randn_like(p)
                p = p.clamp(0, 1)
            cx, cy = int((x1 + x2) / 2), int(y1 + 0.4 * (y2 - y1))
            px, py = max(0, cx - side // 2), max(0, cy - side // 2)
            px2, py2 = min(S, px + side), min(S, py + side)
            x[:, :, py:py2, px:px2] = p[:, :, : py2 - py, : px2 - px]
        return x

    def person_score(x):
        out = net(x); out = out[0] if isinstance(out, (list, tuple)) else out
        return out[:, 4, :]                     # class 0 (person) score of every anchor

    def tv(p): return (p[:, :, 1:] - p[:, :, :-1]).abs().mean() + (p[:, 1:, :] - p[:, :-1, :]).abs().mean()

    # ---- reference detections (clean YOLO output acts as ground truth) ----
    ref_tr = [detect(i) for i in train]
    ref_te = [detect(i) for i in test]

    # ---- 1) learn the universal sticker ----
    patch = torch.rand(1, 3, 128, 128, device=dev, requires_grad=True)
    opt = torch.optim.Adam([patch], lr=0.03)
    for ep in range(a.epochs):
        tot = 0.0; cnt = 0
        for img, ref in zip(train, ref_tr):
            if not ref: continue
            x = place(to_t(img), ref, patch, eot=True)
            loss = person_score(x).max() + 2.5 * tv(patch[0])
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): patch.clamp_(0, 1)
            tot += loss.item(); cnt += 1
        print(f"epoch {ep+1}/{a.epochs} loss={tot/max(cnt,1):.4f}")
    patch = patch.detach()
    Image.fromarray((patch[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f"{a.out}/patch.png")

    # ---- 2) evaluate on held-out images ----
    cond = ["clean", "random", "attacked", "def_clean", "def_attacked"]
    C = {c: dict(tp=0, fn=0, fp=0) for c in cond}      # micro-averaged confusion counts
    alerts_att = alerts_clean = n_img = 0
    t_det, t_def = [], []

    def tally(c, ref, det, thr=0.5):
        tp = sum(any(iou(r, d) >= thr for d in det) for r in ref)
        fp = sum(not any(iou(r, d) >= thr for r in ref) for d in det)
        C[c]["tp"] += tp; C[c]["fn"] += len(ref) - tp; C[c]["fp"] += fp

    for img, ref in zip(test, ref_te):
        if not ref: continue
        att = (place(to_t(img), ref, patch).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rnd = (place(to_t(img), ref, torch.rand_like(patch)).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        t0 = time.perf_counter(); d_att = detect(att); t_det.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); san, _ = patchguard(att, win=a.win, k=a.k); d_def_att = detect(san); t_def.append(time.perf_counter() - t0)
        san_c, _ = patchguard(img, win=a.win, k=a.k); d_def_c = detect(san_c); d_c = detect(img)
        tally("clean", ref, d_c); tally("random", ref, detect(rnd)); tally("attacked", ref, d_att)
        tally("def_clean", ref, d_def_c); tally("def_attacked", ref, d_def_att)
        n_img += 1
        alerts_att += len(d_def_att) > len(d_att)
        alerts_clean += len(d_def_c) > len(d_c)

    res = {"test_images_used": n_img, "counts": C}
    for c in cond:
        tp, fn, fp = C[c]["tp"], C[c]["fn"], C[c]["fp"]
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        res[c] = dict(precision=p, recall=r, f1=(2 * p * r / (p + r) if p + r else 0.0))
    res["attack_success_rate"] = 1 - res["attacked"]["recall"]
    res["attack_success_rate_after_defence"] = 1 - res["def_attacked"]["recall"]
    res["alert_rate_attacked"] = alerts_att / max(n_img, 1)
    res["false_alert_rate_clean"] = alerts_clean / max(n_img, 1)
    res["latency_ms_detector_only"] = 1000 * float(np.mean(t_det))
    res["latency_ms_patchguard_pipeline"] = 1000 * float(np.mean(t_def))
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = ["Clean", "Random\nsticker", "Adversarial\nsticker", "PatchGuard\n(clean)", "PatchGuard\n(adversarial)"]
    vals = [res[c]["recall"] for c in cond]
    plt.figure(figsize=(7, 4)); plt.bar(names, vals, color=["#2E7D32", "#9E9E9E", "#C62828", "#1565C0", "#EF6C00"])
    plt.ylabel("Person recall @IoU 0.5"); plt.ylim(0, 1.05); plt.title("Effect of sticker attack and PatchGuard defence")
    plt.tight_layout(); plt.savefig(f"{a.out}/recall_chart.png", dpi=200)

    # qualitative grid for the first test image that has a person
    for img, ref in zip(test, ref_te):
        if ref:
            att = (place(to_t(img), ref, patch).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            san, m = patchguard(att, win=a.win, k=a.k)
            fig, ax = plt.subplots(1, 4, figsize=(14, 4))
            for axi, im, t in zip(ax, [img, att, (m * 255).astype(np.uint8), san], ["Clean", "With sticker", "Patch mask M", "Sanitised"]):
                axi.imshow(im, cmap="gray" if im.ndim == 2 else None); axi.set_title(t); axi.axis("off")
            plt.tight_layout(); plt.savefig(f"{a.out}/qualitative.png", dpi=200); break

if __name__ == "__main__":
    main()
