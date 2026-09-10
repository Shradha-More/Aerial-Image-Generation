import os
import cv2
import numpy as np
from scipy import fftpack

# === CONFIG ===
IMG_PATH = "input.jpg"
OUTDIR = "outputs"
METADATA = {
    "aircraft_speed_m_s": 220,
    "shutter_s": 0.00125,
    "object_distance_m": 420,
    "focal_length_mm": 50,
    "sensor_width_mm": 36,
    "cam_angle_deg": 8
}
WIENER_K = 0.01

# === HELPERS ===
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def compute_psf(aircraft_speed_m_s, shutter_s, object_distance_m,
                focal_length_mm, sensor_width_mm, image_width_px,
                cam_angle_deg=0, psf_len_px=None):
    L_m = aircraft_speed_m_s * shutter_s
    shift_sensor_mm = (L_m * focal_length_mm) / (object_distance_m + 1e-9)
    mm_per_pixel = sensor_width_mm / float(image_width_px)
    shift_px = shift_sensor_mm / (mm_per_pixel + 1e-9)

    if psf_len_px is None:
        psf_len_px = max(3, int(round(abs(shift_px) + 1)))

    psf = np.zeros((psf_len_px, psf_len_px), dtype=np.float32)
    center = psf_len_px // 2
    psf[center, :] = 1.0

    M = cv2.getRotationMatrix2D((center, center), cam_angle_deg, 1.0)
    psf = cv2.warpAffine(psf, M, (psf_len_px, psf_len_px), flags=cv2.INTER_LINEAR)
    psf /= psf.sum()
    return psf

def wiener_deconv_channel(channel, psf, K):
    img_fft = fftpack.fft2(channel)
    ph, pw = psf.shape
    psf_padded = np.zeros_like(channel)
    psf_padded[:ph, :pw] = psf
    psf_fft = fftpack.fft2(psf_padded)
    psf_fft_conj = np.conj(psf_fft)
    denom = (psf_fft * psf_fft_conj) + K
    wiener_filter = psf_fft_conj / denom
    res_fft = img_fft * wiener_filter
    res = np.real(fftpack.ifft2(res_fft))
    return np.clip(res, 0.0, 1.0)

def wiener_deconv_color(img_bgr, psf, K):
    img = img_bgr.astype(np.float32) / 255.0
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = wiener_deconv_channel(img[:, :, c], psf, K)
    return (out * 255.0).astype(np.uint8)

# === MAIN ===
if not os.path.isfile(IMG_PATH):
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")

ensure_dir(OUTDIR)
print(f"[INFO] Loading {IMG_PATH}...")
img = cv2.imread(IMG_PATH)
print("[INFO] Image shape:", img.shape)

print("[INFO] Computing PSF...")
psf = compute_psf(
    METADATA["aircraft_speed_m_s"],
    METADATA["shutter_s"],
    METADATA["object_distance_m"],
    METADATA["focal_length_mm"],
    METADATA["sensor_width_mm"],
    img.shape[1],
    cam_angle_deg=METADATA["cam_angle_deg"]
)
print("[INFO] PSF shape:", psf.shape)

print("[INFO] Running Wiener deblurring...")
deblurred = wiener_deconv_color(img, psf, WIENER_K)

out_path = os.path.join(OUTDIR, "deblurred.png")
cv2.imwrite(out_path, deblurred)
print(f"[INFO] Saved deblurred image to {out_path}")
