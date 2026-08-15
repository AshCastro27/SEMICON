"""
generate_dataset.py  — DriftSense-AI Standalone Dataset Generator
I4C SEMICON India Hackathon 2026 · Problem Statement 2

Generates synthetic semiconductor image pairs (Reference + Search) for
DRAM and FinFET architectures with realistic SEM-inspired degradation.

Usage:
    python generate_dataset.py --arch DRAM   --num 30 --out ./dataset_dram
    python generate_dataset.py --arch FINFET --num 30 --out ./dataset_finfet
    python generate_dataset.py --arch BOTH   --num 60 --out ./dataset --seed 42

Output structure:
    <out>/
      train/
        pair_000000/
          reference.png        <- degraded reference crop
          search.png           <- larger search image
          ground_truth.json    <- true center_x, center_y + metadata
        pair_000001/ ...
      validation/  ...
      test/        ...

Ground truth format (ground_truth.json):
    {
      "pair_id":    "pair_000000",
      "arch":       "DRAM",
      "center_x":   412,         <- true location of ref center in search
      "center_y":   317,
      "search_width":  1000,
      "search_height": 1000,
      "ref_width":   96,
      "ref_height":  96,
      "rotation_deg": 1.23,
      "scale":        0.97,
      "noise_sigma":  8.4,
      "blur_type":   "motion",
      "seed":        42000
    }

Requirements: numpy, opencv-python, tqdm
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):          # silent fallback
        return it


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIG (inlined — no external config.yaml required)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    "search_size": 1000,
    "reference_min": 64,
    "reference_max": 128,
    "dataset": {
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
    },
    "geometry": {
        "rotation_max_deg": 5.0,
        "scale_jitter_min": 0.92,
        "scale_jitter_max": 1.08,
    },
    "noise": {
        "gaussian_sigma_min": 3.0,
        "gaussian_sigma_max": 12.0,
        "speckle_strength_min": 0.02,
        "speckle_strength_max": 0.10,
        "salt_pepper_prob_min": 0.001,
        "salt_pepper_prob_max": 0.008,
        "search_extra_sigma_min": 2.0,
        "search_extra_sigma_max": 8.0,
    },
    "blur": {
        "gaussian_sigma_min": 0.3,
        "gaussian_sigma_max": 1.5,
        "motion_blur_max_len": 9,
        "defocus_sigma_max": 1.5,
    },
    "edge_brightening": {
        "strength_min": 0.05,
        "strength_max": 0.25,
        "sigma": 1.0,
    },
    "dram": {
        "canvas_min": 1100,
        "canvas_max": 1200,
        "base_intensity_min": 30,
        "base_intensity_max": 60,
        "wordline_pitch_min": 14,
        "wordline_pitch_max": 22,
        "bitline_pitch_min": 14,
        "bitline_pitch_max": 22,
        "line_width_fraction": 0.35,
        "line_intensity_min": 160,
        "line_intensity_max": 220,
        "contact_radius_min": 1,
        "contact_radius_max": 3,
        "contact_intensity": 240,
        "defect_prob": 0.02,
    },
    "finfet": {
        "canvas_min": 1100,
        "canvas_max": 1200,
        "fin_pitch_min": 10,
        "fin_pitch_max": 18,
        "fin_width_fraction": 0.30,
        "fin_intensity_min": 150,
        "fin_intensity_max": 200,
        "gate_pitch_min": 28,
        "gate_pitch_max": 48,
        "gate_width_fraction": 0.28,
        "gate_intensity_min": 170,
        "gate_intensity_max": 220,
        "intersection_boost": 35,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# DRAM GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _generate_dram(rng, cfg, canvas_size):
    """Synthesise a DRAM-like grayscale image (word-lines + bit-lines + contacts)."""
    c = cfg["dram"]
    H = W = canvas_size
    base = int(rng.integers(c["base_intensity_min"], c["base_intensity_max"] + 1))
    img  = np.full((H, W), base, dtype=np.float32)

    # Word-lines (horizontal)
    wl_pitch = int(rng.integers(c["wordline_pitch_min"], c["wordline_pitch_max"] + 1))
    wl_width = max(1, int(wl_pitch * c["line_width_fraction"]))
    wl_int   = int(rng.integers(c["line_intensity_min"],  c["line_intensity_max"]  + 1))
    for y in range(0, H, wl_pitch):
        row_var = int(rng.integers(-8, 9))
        row_int = int(np.clip(wl_int + row_var, 0, 255))
        y0, y1  = y, min(H, y + wl_width)
        img[y0:y1, :] = row_int
        if rng.random() < c["defect_prob"]:
            xs = int(rng.integers(0, W - 10)); xl = int(rng.integers(5, 20))
            img[y0:y1, xs:xs+xl] = base

    # Bit-lines (vertical)
    bl_pitch = int(rng.integers(c["bitline_pitch_min"], c["bitline_pitch_max"] + 1))
    bl_width = max(1, int(bl_pitch * c["line_width_fraction"]))
    bl_int   = int(rng.integers(c["line_intensity_min"], c["line_intensity_max"] + 1))
    for x in range(0, W, bl_pitch):
        col_var = int(rng.integers(-8, 9))
        col_int = int(np.clip(bl_int + col_var, 0, 255))
        x0, x1  = x, min(W, x + bl_width)
        img[:, x0:x1] = np.maximum(img[:, x0:x1], col_int)
        if rng.random() < c["defect_prob"]:
            ys = int(rng.integers(0, H - 10)); yl = int(rng.integers(5, 20))
            img[ys:ys+yl, x0:x1] = base

    # Contacts / vias
    for y in range(0, H, wl_pitch):
        for x in range(0, W, bl_pitch):
            if rng.random() < 0.85:
                r = int(rng.integers(c["contact_radius_min"], c["contact_radius_max"] + 1))
                cv2.circle(img, (x, y), r, float(c["contact_intensity"]), -1)

    # Illumination non-uniformity
    gx = np.linspace(1.0, float(rng.uniform(0.85, 1.15)), W)
    gy = np.linspace(1.0, float(rng.uniform(0.85, 1.15)), H)
    img = np.clip(img * np.outer(gy, gx), 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# FINFET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _generate_finfet(rng, cfg, canvas_size):
    """Synthesise a FinFET-like grayscale image (fins + gates + intersections)."""
    c = cfg["finfet"]
    H = W = canvas_size
    img = np.zeros((H, W), dtype=np.float32)

    # Fins (narrow vertical stripes)
    fin_pitch = int(rng.integers(c["fin_pitch_min"], c["fin_pitch_max"] + 1))
    fin_width = max(1, int(fin_pitch * c["fin_width_fraction"]))
    fin_int   = int(rng.integers(c["fin_intensity_min"], c["fin_intensity_max"] + 1))
    fin_pos   = []
    for x in range(0, W, fin_pitch):
        jitter = int(rng.integers(-1, 2))
        xj = int(np.clip(x + jitter, 0, W - 1))
        x0, x1 = xj, min(W, xj + fin_width)
        fin_pos.append((x0, x1))
        col_int = int(np.clip(fin_int + int(rng.integers(-10, 11)), 0, 255))
        img[:, x0:x1] = col_int

    # Gates (horizontal, wider pitch)
    gate_pitch = int(rng.integers(c["gate_pitch_min"], c["gate_pitch_max"] + 1))
    gate_width = max(2, int(gate_pitch * c["gate_width_fraction"]))
    gate_int   = int(rng.integers(c["gate_intensity_min"], c["gate_intensity_max"] + 1))
    gate_pos   = []
    for y in range(0, H, gate_pitch):
        jitter = int(rng.integers(-1, 2))
        yj = int(np.clip(y + jitter, 0, H - 1))
        y0, y1 = yj, min(H, yj + gate_width)
        gate_pos.append((y0, y1))
        row_int = int(np.clip(gate_int + int(rng.integers(-10, 11)), 0, 255))
        img[y0:y1, :] = np.maximum(img[y0:y1, :], row_int)

    # Gate-fin intersections (brighter in SEM)
    boost = c["intersection_boost"]
    for (x0, x1) in fin_pos:
        for (y0, y1) in gate_pos:
            img[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1].astype(np.float32) + boost, 0, 255)

    # Spacer halo
    for y0, y1 in gate_pos:
        for r in range(max(0, y0 - 1), min(H, y1 + 1)):
            if r < y0 or r >= y1:
                img[r, :] = np.clip(img[r, :] - 15, 0, 255)

    # Illumination non-uniformity
    gx = np.linspace(1.0, float(rng.uniform(0.88, 1.12)), W)
    gy = np.linspace(1.0, float(rng.uniform(0.88, 1.12)), H)
    img = np.clip(img * np.outer(gy, gx), 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# DEGRADATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _add_gaussian_noise(img, rng, sigma):
    if sigma <= 0: return img
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

def _add_speckle(img, rng, strength):
    if strength <= 0: return img
    return np.clip(img.astype(np.float32) * (1 + rng.normal(0, strength, img.shape)), 0, 255).astype(np.uint8)

def _add_salt_pepper(img, rng, prob):
    if prob <= 0: return img
    out  = img.copy(); mask = rng.random(img.shape)
    out[mask < prob / 2] = 0; out[mask > 1 - prob / 2] = 255
    return out

def _intensity_variation(img, rng):
    scale = rng.uniform(0.85, 1.15); shift = rng.uniform(-15, 15)
    return np.clip(img.astype(np.float32) * scale + shift, 0, 255).astype(np.uint8)

def _gaussian_blur(img, sigma):
    if sigma <= 0: return img
    ks = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(img, (ks, ks), sigma)

def _motion_blur(img, rng, max_len):
    length = int(rng.integers(3, max_len + 1)) | 1
    angle  = float(rng.uniform(0, 360))
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    cv2.ellipse(kernel, (center, center), (center, 0), angle, 0, 360, 1, -1)
    s = kernel.sum()
    if s > 0: kernel /= s
    return cv2.filter2D(img, -1, kernel)

def _edge_brightening(img, rng, cfg):
    ec = cfg["edge_brightening"]
    strength = float(rng.uniform(ec["strength_min"], ec["strength_max"]))
    if strength <= 0: return img
    blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), ec["sigma"])
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    if mag.max() > 0: mag = mag / mag.max() * 255.0
    return np.clip(img.astype(np.float32) + strength * mag, 0, 255).astype(np.uint8)

def _degrade(img, rng, cfg, extra_noise=0.0, is_search=False):
    nc, bc = cfg["noise"], cfg["blur"]
    img = _edge_brightening(img, rng, cfg)
    bt  = int(rng.integers(0, 3))
    if bt == 0:
        img = _gaussian_blur(img, float(rng.uniform(bc["gaussian_sigma_min"], bc["gaussian_sigma_max"])))
    elif bt == 1:
        img = _motion_blur(img, rng, bc["motion_blur_max_len"])
    else:
        img = _gaussian_blur(img, float(rng.uniform(0.3, bc["defocus_sigma_max"])))
    img = _intensity_variation(img, rng)
    img = _add_gaussian_noise(img, rng, float(rng.uniform(nc["gaussian_sigma_min"], nc["gaussian_sigma_max"])))
    img = _add_speckle(img, rng, float(rng.uniform(nc["speckle_strength_min"], nc["speckle_strength_max"])))
    img = _add_salt_pepper(img, rng, float(rng.uniform(nc["salt_pepper_prob_min"], nc["salt_pepper_prob_max"])))
    if is_search and extra_noise > 0:
        img = _add_gaussian_noise(img, rng, extra_noise)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# PLACEMENT & JITTER
# ─────────────────────────────────────────────────────────────────────────────

def _random_placement(rng, search_size, ref_h, ref_w, margin=10):
    half_h = ref_h // 2 + margin
    half_w = ref_w // 2 + margin
    x_min, x_max = half_w, search_size - half_w
    y_min, y_max = half_h, search_size - half_h
    if x_max <= x_min or y_max <= y_min:
        return search_size // 2, search_size // 2
    return int(rng.integers(x_min, x_max + 1)), int(rng.integers(y_min, y_max + 1))

def _apply_jitter(img, rng, cfg):
    geo = cfg["geometry"]
    h, w = img.shape[:2]
    angle = float(rng.uniform(-geo["rotation_max_deg"], geo["rotation_max_deg"]))
    scale = float(rng.uniform(geo["scale_jitter_min"],  geo["scale_jitter_max"]))
    M     = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return warped, angle, scale


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE PAIR GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _generate_pair(pair_idx, architecture, cfg, split, base_seed):
    pair_seed  = base_seed + pair_idx * 1000
    rng        = np.random.default_rng(pair_seed)

    search_size = cfg["search_size"]
    ref_min     = cfg["reference_min"]
    ref_max     = cfg["reference_max"]
    canvas_size = search_size + 64

    arch = architecture
    if arch == "BOTH":
        arch = "DRAM" if rng.random() < 0.5 else "FINFET"

    # Generate large canvas
    if arch == "DRAM":
        canvas = _generate_dram(rng, cfg, canvas_size)
    else:
        canvas = _generate_finfet(rng, cfg, canvas_size)

    # Reference size
    ref_h = int(rng.integers(ref_min, ref_max + 1))
    ref_w = int(rng.integers(ref_min, ref_max + 1))

    # Random target location
    cx, cy = _random_placement(rng, search_size, ref_h, ref_w, margin=ref_max // 2)

    # Extract reference crop
    x0 = int(np.clip(cx - ref_w // 2, 0, canvas_size - ref_w))
    y0 = int(np.clip(cy - ref_h // 2, 0, canvas_size - ref_h))
    ref_crop = canvas[y0:y0 + ref_h, x0:x0 + ref_w].copy()

    # Geometric jitter on reference
    ref_jittered, rot_deg, scale = _apply_jitter(ref_crop, np.random.default_rng(pair_seed + 1), cfg)

    # Extra search noise
    extra_sigma = float(np.random.default_rng(pair_seed + 2).uniform(
        cfg["noise"]["search_extra_sigma_min"], cfg["noise"]["search_extra_sigma_max"]))

    # Degrade reference (lighter)
    ref_img = _degrade(ref_jittered, np.random.default_rng(pair_seed + 2), cfg,
                       extra_noise=0.0, is_search=False)

    # Degrade search image
    search_canvas = canvas[0:search_size, 0:search_size].copy()
    search_img    = _degrade(search_canvas, np.random.default_rng(pair_seed + 3), cfg,
                             extra_noise=extra_sigma, is_search=True)

    blur_types = ["gaussian", "motion", "defocus"]
    blur_type  = blur_types[int(np.random.default_rng(pair_seed + 2).integers(0, 3))]

    gt = {
        "pair_id":       f"pair_{pair_idx:06d}",
        "arch":          arch,
        "split":         split,
        "center_x":      int(cx),
        "center_y":      int(cy),
        "search_width":  search_size,
        "search_height": search_size,
        "ref_width":     ref_w,
        "ref_height":    ref_h,
        "rotation_deg":  round(float(rot_deg), 4),
        "scale":         round(float(scale), 4),
        "noise_sigma":   round(float(extra_sigma), 4),
        "blur_type":     blur_type,
        "seed":          int(pair_seed),
    }
    return ref_img, search_img, gt


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DriftSense-AI: Synthetic Semiconductor Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arch",  choices=["DRAM", "FINFET", "BOTH"], default="BOTH",
                        help="Architecture style to generate")
    parser.add_argument("--num",   type=int, default=30,
                        help="Total number of image pairs to generate")
    parser.add_argument("--out",   type=str, default="./dataset",
                        help="Output directory")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Global random seed for reproducibility")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all",
                        help="Which split(s) to generate")
    args = parser.parse_args()

    cfg = DEFAULT_CFG
    n   = args.num
    ds  = cfg["dataset"]

    if args.split == "all":
        n_train = max(1, int(n * ds["train_fraction"]))
        n_val   = max(1, int(n * ds["validation_fraction"]))
        n_test  = max(1, n - n_train - n_val)
        splits  = {
            "train":      list(range(0, n_train)),
            "validation": list(range(n_train, n_train + n_val)),
            "test":       list(range(n_train + n_val, n_train + n_val + n_test)),
        }
    else:
        splits = {args.split: list(range(n))}

    print(f"\nDriftSense-AI Dataset Generator")
    print(f"  Architecture : {args.arch}")
    print(f"  Total pairs  : {n}")
    print(f"  Output dir   : {args.out}")
    print(f"  Seed         : {args.seed}\n")

    for split_name, indices in splits.items():
        print(f"Generating [{split_name}] — {len(indices)} pairs ...")
        for pair_idx in tqdm(indices, unit="pair"):
            ref_img, search_img, gt = _generate_pair(
                pair_idx, args.arch, cfg, split_name, args.seed
            )
            pair_dir = os.path.join(args.out, split_name, gt["pair_id"])
            os.makedirs(pair_dir, exist_ok=True)
            cv2.imwrite(os.path.join(pair_dir, "reference.png"), ref_img)
            cv2.imwrite(os.path.join(pair_dir, "search.png"),    search_img)
            with open(os.path.join(pair_dir, "ground_truth.json"), "w") as f:
                json.dump(gt, f, indent=2)

    print(f"\nDone. Dataset saved to: {args.out}")


if __name__ == "__main__":
    main()
