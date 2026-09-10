import os, sys
import numpy as np
import cv2
import torch

PROJECT_ROOT = r"D:\Desktop\Shraddha\DRDO-Aerial-Image-Generation"
MIDAS_REPO   = os.path.join(PROJECT_ROOT, "MiDaS")
IN_IMG       = os.path.join(PROJECT_ROOT, "outputs", "deblurred.png")
OUT_DIR      = os.path.join(PROJECT_ROOT, "outputs")
OUT_NPY      = os.path.join(OUT_DIR, "depth_norm.npy")
OUT_PREVIEW  = os.path.join(OUT_DIR, "depth_preview.png")

device = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_image_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def run_midas_depth(image_rgb, model_variant="DPT_Hybrid"):
    if os.path.isdir(MIDAS_REPO) and MIDAS_REPO not in sys.path:
        sys.path.append(MIDAS_REPO)

    midas = torch.hub.load('intel-isl/MiDaS', model_variant).to(device).eval()
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

    if model_variant in ["DPT_Large", "DPT_Hybrid", "DPT_BEiT_L_512"]:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    inp = transform(image_rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    dmin, dmax = float(pred.min()), float(pred.max())
    depth_norm = (pred - dmin) / (dmax - dmin + 1e-9)
    return depth_norm

def save_preview(depth_norm, path_png):
    depth_u8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path_png, depth_u8)

def main():
    ensure_dir(OUT_DIR)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading {IN_IMG}")
    img_rgb = load_image_rgb(IN_IMG)

    print("[INFO] Running MiDaS depth (DPT_Hybrid)...")
    depth_norm = run_midas_depth(img_rgb, model_variant="DPT_Hybrid")
    np.save(OUT_NPY, depth_norm)
    save_preview(depth_norm, OUT_PREVIEW)

    print("\n=== Depth step complete ===")
    print(f"Depth (normalized) saved: {OUT_NPY}")
    print(f"Depth preview (8-bit PNG): {OUT_PREVIEW}")

if __name__ == "__main__":
    main()
