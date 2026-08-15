"""
inference.py  -- DriftSense-AI Localization Inference
I4C SEMICON India Hackathon 2026 · Problem Statement 2

STANDALONE: No external config file or local module imports required.
All localization logic is inlined here.

Usage (competition-safe):
    python inference.py reference.png search.png

Output:
    x y

Optional flags:
    python inference.py reference.png search.png --json     # JSON output
    python inference.py reference.png search.png --verbose  # Debug to stderr
    python inference.py reference.png search.png --arch FINFET  # Override arch

DL model weights (optional, auto-loaded if present):
    finfet_siamese_cpu.pth   <- place in same directory as this script

Requirements: numpy, opencv-python, scipy
Optional:     torch  (only for Siamese CNN on FinFET images)
"""

import sys, os, argparse, json, time
import numpy as np
import cv2
from dataclasses import dataclass
from scipy.signal import argrelmax

# ─────────────────────────────────────────────────────────────────────────────
# INLINED CONFIG (no config.yaml dependency)
# ─────────────────────────────────────────────────────────────────────────────

_CFG = {
    "top_k": 10,
    "nms_radius": 20,
    "ncc_margin_threshold": 0.001,
    "rotation_confidence_threshold": 0.0,
    "rotations_deg": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    "scales": [0.90, 0.925, 0.95, 0.975, 1.00, 1.025, 1.05, 1.075, 1.10],
    "preprocessing": {
        "gaussian_blur_sigma": 0.5,
        "use_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_tile_size": 8,
    },
    "weights": {
        "appearance":   0.50,
        "edge":         0.30,
        "neighborhood": 0.12,
        "periodicity":  0.06,
        "scale":        0.02,
    },
    "refinement": {
        "search_radius": 15,
        "subpixel": False,
    },
    "periodicity": {
        "autocorr_window": 64,
        "fft_peak_count": 5,
        "consistency_radius": 50,
        "expand_factor": 3.0,
    },
    "arch_detect": {
        "enabled": True,
        "anisotropy_threshold": 1.4,
        "col_dominance_threshold": 1.3,
        "confidence_threshold": 0.20,
    },
    "finfet_search": {
        "w_appearance": 0.35,
        "w_edge": 0.65,
        "nms_radius": 10,
        "top_k": 20,
        "scales": [0.90, 0.925, 0.95, 0.975, 1.00, 1.025, 1.05, 1.075, 1.10],
        "pc_expand_factor": 2.0,
    },
    "finfet_weights": {
        "appearance":   0.20,
        "edge":         0.30,
        "neighborhood": 0.45,
        "periodicity":  0.03,
        "scale":        0.02,
    },
    "siamese": {
        "patch_size": 64,
        "enabled": True,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(img: np.ndarray, cfg: dict) -> dict:
    pc = cfg
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = pc.get("gaussian_blur_sigma", 0)
    gray_smooth = cv2.GaussianBlur(gray, (int(6*sigma+1)|1, int(6*sigma+1)|1), sigma) if sigma > 0 else gray
    if pc.get("use_clahe", True):
        clahe = cv2.createCLAHE(clipLimit=pc.get("clahe_clip_limit", 2.0),
                                 tileGridSize=(pc.get("clahe_tile_size", 8),)*2)
        clahe_img = clahe.apply(gray_smooth)
    else:
        clahe_img = gray_smooth
    f = clahe_img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    sobel = (mag / mag.max()) if mag.max() > 0 else mag
    return {"gray": gray, "gray_smooth": gray_smooth, "clahe": clahe_img, "sobel": sobel}


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    x: int = 0; y: int = 0; scale: float = 1.0
    appearance_score: float = 0.0; edge_score: float = 0.0
    neighborhood_score: float = 0.0; periodicity_score: float = 0.0
    final_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SCALE NCC SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def _ncc_response(search, tmpl):
    return cv2.matchTemplate(search.astype(np.float32), tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)

def _combined_response(s_gray, s_edge, r_gray, r_edge, w_app, w_edg):
    app_map  = np.clip(_ncc_response(s_gray, r_gray), 0, 1)
    se = (s_edge * 255).astype(np.float32); re = (r_edge * 255).astype(np.float32)
    edge_map = np.clip(_ncc_response(se, re), 0, 1)
    h = min(app_map.shape[0], edge_map.shape[0]); w = min(app_map.shape[1], edge_map.shape[1])
    return w_app * app_map[:h,:w] + w_edg * edge_map[:h,:w], app_map[:h,:w], edge_map[:h,:w]

def _nms(resp, radius, top_k):
    r = resp.copy(); peaks = []
    for _ in range(top_k * 5):
        idx = np.argmax(r); score = r.flat[idx]
        if score <= 0: break
        py, px = np.unravel_index(idx, r.shape)
        peaks.append((float(score), int(py), int(px)))
        y0,y1 = max(0,py-radius), min(r.shape[0],py+radius+1)
        x0,x1 = max(0,px-radius), min(r.shape[1],px+radius+1)
        r[y0:y1,x0:x1] = -1.0
        if len(peaks) >= top_k: break
    return peaks

def _multiscale_search(ref_pre, srch_pre, scales, top_k, nms_r, w_app, w_edg, rotations=None):
    r_gray = ref_pre.get("gray_smooth", ref_pre["clahe"]).astype(np.float32)
    r_edge = ref_pre["sobel"]
    s_gray = srch_pre.get("gray_smooth", srch_pre["clahe"]).astype(np.float32)
    s_edge = srch_pre["sobel"]
    rh, rw = r_gray.shape
    rotations = rotations or [0.0]
    cands, maps = [], {}
    for scale in scales:
        nh = max(8, int(rh * scale)); nw = max(8, int(rw * scale))
        if nh >= s_gray.shape[0] or nw >= s_gray.shape[1]: continue
        rgs = cv2.resize(r_gray, (nw, nh)); res = cv2.resize(r_edge, (nw, nh))
        best_c = best_a = best_e = None
        for ang in rotations:
            rg, re = rgs, res
            if abs(ang) > 0.05:
                M = cv2.getRotationMatrix2D((nw/2, nh/2), ang, 1.0)
                rg = cv2.warpAffine(rgs, M, (nw,nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                re = cv2.warpAffine(res, M, (nw,nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            c, a, e = _combined_response(s_gray, s_edge, rg, re, w_app, w_edg)
            if best_c is None: best_c, best_a, best_e = c.copy(), a.copy(), e.copy()
            else:
                h2 = min(best_c.shape[0],c.shape[0]); w2 = min(best_c.shape[1],c.shape[1])
                mask = c[:h2,:w2] > best_c[:h2,:w2]
                best_c[:h2,:w2] = np.where(mask, c[:h2,:w2], best_c[:h2,:w2])
                best_a[:h2,:w2] = np.where(mask, a[:h2,:w2], best_a[:h2,:w2])
                best_e[:h2,:w2] = np.where(mask, e[:h2,:w2], best_e[:h2,:w2])
        if best_c is None: continue
        maps[scale] = best_c
        for score, py, px in _nms(best_c, nms_r, top_k):
            cx = px + nw // 2; cy = py + nh // 2
            a = float(best_a[py,px]) if py < best_a.shape[0] and px < best_a.shape[1] else score
            e = float(best_e[py,px]) if py < best_e.shape[0] and px < best_e.shape[1] else 0.0
            cands.append(Candidate(x=cx, y=cy, scale=scale, appearance_score=a, edge_score=e))
    cands.sort(key=lambda c: c.appearance_score * w_app + c.edge_score * w_edg, reverse=True)
    return cands, maps


# ─────────────────────────────────────────────────────────────────────────────
# PERIODICITY SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _fft_desc(img, n=5):
    F = np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float32))))
    cy, cx = F.shape[0]//2, F.shape[1]//2; F[cy-2:cy+3, cx-2:cx+3] = 0
    flat = F.flatten(); top = np.sort(flat[np.argpartition(flat,-n)[-n:]])[::-1]
    return (top / top[0]) if top[0] > 0 else top

def _fft_sim(d1, d2):
    n1,n2 = np.linalg.norm(d1),np.linalg.norm(d2)
    return float(np.dot(d1,d2)/(n1*n2)) if n1>1e-9 and n2>1e-9 else 0.0

def _period_autocorr(sig, min_p=4):
    n = len(sig)
    if n < 2*min_p: return 0.0
    s = sig - sig.mean()
    if np.abs(s).max() < 1e-6: return 0.0
    acf = np.correlate(s, s, mode="full")[n-1:]; acf /= (acf[0]+1e-9)
    peaks = argrelmax(acf[min_p:], order=2)[0]
    if len(peaks) == 0: return 0.0
    return float(peaks[np.argmax(acf[min_p:][peaks])] + min_p)

def _score_periodicity(cands, ref_pre, srch_pre, cfg):
    pc = cfg["periodicity"]
    ref_img = ref_pre["clahe"]; srch_img = srch_pre["clahe"]
    ref_desc = _fft_desc(ref_img, pc["fft_peak_count"])
    for c in cands:
        r = pc["consistency_radius"]
        x0,y0 = c.x-r, c.y-r; x1,y1 = c.x+r, c.y+r
        h,w = srch_img.shape
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            c.neighborhood_score = 0.5; c.periodicity_score = 0.5; continue
        neigh = srch_img[y0:y1,x0:x1]
        # Period similarity
        ref_rp  = _period_autocorr(ref_img.astype(np.float32).mean(axis=1))
        ref_cp  = _period_autocorr(ref_img.astype(np.float32).mean(axis=0))
        cand_rp = _period_autocorr(neigh.astype(np.float32).mean(axis=1))
        cand_cp = _period_autocorr(neigh.astype(np.float32).mean(axis=0))
        def psim(p1,p2):
            if p1<=0 or p2<=0: return 0.5
            return float(min(p1,p2)/max(p1,p2))
        fft_s = _fft_sim(ref_desc, _fft_desc(neigh, pc["fft_peak_count"]))
        c.neighborhood_score = 0.25*psim(ref_rp,cand_rp) + 0.25*psim(ref_cp,cand_cp) + 0.50*fft_s
        ref_h,ref_w = ref_img.shape
        rr = max(ref_h,ref_w)//2
        cx,cy = c.x,c.y
        px0,py0 = cx-rr, cy-rr; px1,py1 = cx+rr, cy+rr
        if px0>=0 and py0>=0 and px1<w and py1<h:
            patch = srch_img[py0:py1,px0:px1]
            c.periodicity_score = _fft_sim(ref_desc, _fft_desc(patch, pc["fft_peak_count"]))
        else:
            c.periodicity_score = 0.5
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT SCORING (DRAM path disambiguation)
# ─────────────────────────────────────────────────────────────────────────────

def _ncc_win(tmpl, win):
    if win.shape[0] < tmpl.shape[0] or win.shape[1] < tmpl.shape[1]: return 0.5
    return float(np.clip(cv2.matchTemplate(win.astype(np.float32), tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED).max(), 0, 1))

def _score_with_context(cands, ref_pre, srch_pre, expand_factor=3.0):
    r_gray = ref_pre.get("gray_smooth", ref_pre["clahe"]).astype(np.float32)
    r_edge = (ref_pre["sobel"]*255).astype(np.float32)
    s_gray = srch_pre.get("gray_smooth", srch_pre["clahe"]).astype(np.float32)
    s_edge = (srch_pre["sobel"]*255).astype(np.float32)
    rh,rw = r_gray.shape; sh,sw = s_gray.shape
    half_h = min(max(rh, int(rh*expand_factor/2)), sh//2-1)
    half_w = min(max(rw, int(rw*expand_factor/2)), sw//2-1)
    for c in cands:
        x0,y0 = max(0,c.x-half_w), max(0,c.y-half_h)
        x1,y1 = min(sw,c.x+half_w), min(sh,c.y+half_h)
        c.neighborhood_score = 0.60*_ncc_win(r_gray, s_gray[y0:y1,x0:x1]) + 0.40*_ncc_win(r_edge, s_edge[y0:y1,x0:x1])
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# PHASE CORRELATION (FinFET path)
# ─────────────────────────────────────────────────────────────────────────────

def _pc_score(ref, patch):
    rh,rw = ref.shape; ph,pw = patch.shape
    if ph < rh or pw < rw: return 0.0
    y0,x0 = (ph-rh)//2, (pw-rw)//2
    pc_crop = patch[y0:y0+rh, x0:x0+rw]
    win = np.outer(np.hanning(rh), np.hanning(rw)).astype(np.float32)
    R = np.fft.fft2(ref.astype(np.float32)*win); P = np.fft.fft2(pc_crop.astype(np.float32)*win)
    cross = R * np.conj(P); norm = cross / (np.abs(cross)+1e-9)
    pc_map = np.abs(np.fft.ifft2(norm)).astype(np.float32)
    mean_v = float(pc_map.mean())
    if mean_v < 1e-12: return 0.0
    return float(np.clip(pc_map.max() / (mean_v * max(rh,rw)) / 10.0, 0, 1))

def _score_phase_correlation(cands, ref_pre, srch_pre, expand=2.0):
    r_edge = (ref_pre["sobel"]*255).astype(np.float32)
    r_gray = ref_pre.get("gray_smooth", ref_pre["clahe"]).astype(np.float32)
    s_edge = (srch_pre["sobel"]*255).astype(np.float32)
    s_gray = srch_pre.get("gray_smooth", srch_pre["clahe"]).astype(np.float32)
    rh,rw = r_edge.shape; sh,sw = s_edge.shape
    half_h = min(int(rh*expand), sh//2-1); half_w = min(int(rw*expand), sw//2-1)
    for c in cands:
        y0,y1 = max(0,c.y-half_h), min(sh,c.y+half_h)
        x0,x1 = max(0,c.x-half_w), min(sw,c.x+half_w)
        c.neighborhood_score = 0.65*_pc_score(r_edge, s_edge[y0:y1,x0:x1]) + 0.35*_pc_score(r_gray, s_gray[y0:y1,x0:x1])
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL SIAMESE CNN (FinFET path, requires torch + model weights)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch, torch.nn as nn, torch.nn.functional as F_torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

_SIAMESE_MODEL = None
_SIAMESE_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finfet_siamese_cpu.pth")

class _LightweightEncoder(nn.Module if _TORCH_OK else object):
    def __init__(self):
        if not _TORCH_OK: raise ImportError("PyTorch not available")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.AdaptiveAvgPool2d((4,4)),
        )
        self.fc = nn.Linear(64*16, 128)
    def forward(self, x):
        f = self.encoder(x); f = f.view(f.size(0),-1); return F_torch.normalize(self.fc(f), dim=1)

def _load_siamese():
    global _SIAMESE_MODEL
    if _SIAMESE_MODEL is not None: return _SIAMESE_MODEL
    if not _TORCH_OK or not os.path.isfile(_SIAMESE_WEIGHTS): return None
    try:
        m = _LightweightEncoder()
        ckpt = torch.load(_SIAMESE_WEIGHTS, map_location="cpu", weights_only=True)
        m.load_state_dict(ckpt["model_state"]); m.eval()
        _SIAMESE_MODEL = m; return m
    except Exception: return None

def _siamese_score(cands, ref_img, srch_img, model, patch_size=64):
    if not _TORCH_OK or model is None: return cands
    half = patch_size // 2
    rh,rw = ref_img.shape[:2]
    y0,y1 = max(0,rh//2-half), min(rh,rh//2+half)
    x0,x1 = max(0,rw//2-half), min(rw,rw//2+half)
    rc = ref_img[y0:y1,x0:x1]
    rc = cv2.resize(rc, (patch_size, patch_size)) if (rc.shape[0]<patch_size or rc.shape[1]<patch_size) else cv2.resize(rc, (patch_size, patch_size))
    rf = rc.astype(np.float32); rf = (rf - rf.mean()) / (rf.std()+1e-6)
    ref_t = torch.from_numpy(rf).unsqueeze(0).unsqueeze(0)
    sh, sw = srch_img.shape[:2]
    with torch.no_grad():
        emb_ref = model(ref_t)
        for c in cands:
            py0,py1 = max(0,c.y-half), min(sh,c.y+half)
            px0,px1 = max(0,c.x-half), min(sw,c.x+half)
            patch = srch_img[py0:py1,px0:px1]
            if patch.shape[0] < 4 or patch.shape[1] < 4: c.neighborhood_score = 0.0; continue
            pf = cv2.resize(patch,(patch_size,patch_size)).astype(np.float32); pf=(pf-pf.mean())/(pf.std()+1e-6)
            emb_p = model(torch.from_numpy(pf).unsqueeze(0).unsqueeze(0))
            d = F_torch.pairwise_distance(emb_ref, emb_p, p=2).item()
            c.neighborhood_score = float(max(0.0, 1.0 - d) / 1.0)
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# RANKING
# ─────────────────────────────────────────────────────────────────────────────

def _rank(cands, weights, preferred_scale=1.0):
    if not cands: return cands
    for c in cands:
        c.final_score = float(np.exp(-5.0 * (c.scale - preferred_scale)**2))
    sc_map = {id(c): c.final_score for c in cands}
    for attr in ["neighborhood_score", "periodicity_score"]:
        vals = [getattr(c, attr) for c in cands]; vmin,vmax = min(vals),max(vals)
        if vmax - vmin > 1e-9:
            for c in cands: setattr(c, attr, (getattr(c,attr)-vmin)/(vmax-vmin))
    w = weights
    for c in cands:
        c.final_score = (w["appearance"]*c.appearance_score + w["edge"]*c.edge_score +
                         w["neighborhood"]*c.neighborhood_score + w["periodicity"]*c.periodicity_score +
                         w["scale"]*sc_map[id(c)])
    cands.sort(key=lambda c: c.final_score, reverse=True)
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# ARCH DETECTION (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_arch(img, ad_cfg):
    at  = ad_cfg.get("anisotropy_threshold", 1.4)
    cdt = ad_cfg.get("col_dominance_threshold", 1.3)
    ct  = ad_cfg.get("confidence_threshold", 0.44)
    gray = img.astype(np.float32)
    gx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    aniso = float(np.mean(np.abs(gx))) / (float(np.mean(np.abs(gy)))+1e-6)
    f1 = float(np.clip((aniso-1.0)/max(at-1.0,0.01), 0, 1.5))
    col_p = gray.mean(axis=0); row_p = gray.mean(axis=1)
    def fft_pk(s):
        f = np.abs(np.fft.rfft(s-s.mean())); f[0]=0
        return float(f.max()) if f.max() > 0 else 0.0
    col_dom = fft_pk(col_p) / (fft_pk(row_p)+1e-6)
    f2 = float(np.clip((col_dom-1.0)/max(cdt-1.0,0.01), 0, 1.5))
    score = 0.50*min(f1,1.0) + 0.50*min(f2,1.0)
    if score >= ct: return "FINFET"
    return "DRAM"


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _refine(best, ref_pre, srch_pre, ref_cfg):
    search_radius = ref_cfg.get("search_radius", 15)
    r_gray = ref_pre["clahe"].astype(np.float32)
    s_gray = srch_pre["clahe"].astype(np.float32)
    rh,rw  = r_gray.shape; sh,sw = s_gray.shape
    cx,cy  = best.x, best.y; sc = best.scale
    nh = max(8, int(rh*sc)); nw = max(8, int(rw*sc))
    rg_s = cv2.resize(r_gray, (nw, nh))
    wx0 = max(0, cx-nw//2-search_radius); wy0 = max(0, cy-nh//2-search_radius)
    wx1 = min(sw, cx+nw//2+search_radius+nw); wy1 = min(sh, cy+nh//2+search_radius+nh)
    if (wx1-wx0) < nw or (wy1-wy0) < nh: return cx, cy, best.appearance_score
    win = s_gray[wy0:wy1, wx0:wx1]
    if win.shape[0] < nh or win.shape[1] < nw: return cx, cy, best.appearance_score
    resp = cv2.matchTemplate(win, rg_s, cv2.TM_CCOEFF_NORMED)
    if resp.size == 0: return cx, cy, best.appearance_score
    py,px = np.unravel_index(np.argmax(resp), resp.shape)
    score = float(resp[py,px])
    dx, dy = 0.0, 0.0
    if ref_cfg.get("subpixel", False) and 0 < py < resp.shape[0]-1 and 0 < px < resp.shape[1]-1:
        dx = (resp[py,px-1]-resp[py,px+1]) / (2*(resp[py,px-1]-2*resp[py,px]+resp[py,px+1])+1e-6)
        dy = (resp[py-1,px]-resp[py+1,px]) / (2*(resp[py-1,px]-2*resp[py,px]+resp[py+1,px])+1e-6)
        dx = np.clip(dx, -1.0, 1.0); dy = np.clip(dy, -1.0, 1.0)
    fx = int(np.round(np.clip(wx0+px+dx+nw/2.0, 0, sw-1)))
    fy = int(np.round(np.clip(wy0+py+dy+nh/2.0, 0, sh-1)))
    return fx, fy, score


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _dedup(cands, radius, max_k):
    cands.sort(key=lambda c: c.appearance_score*0.6+c.edge_score*0.4, reverse=True)
    out = []
    for c in cands:
        if not any(np.sqrt((c.x-e.x)**2+(c.y-e.y)**2) < radius for e in out):
            out.append(c)
        if len(out) >= max_k: break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FINFET CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _build_finfet_cands(ref_pre, srch_pre, sh, sw, cfg):
    fs = cfg.get("finfet_search", {})
    w_app = fs.get("w_appearance", 0.35); w_edg = fs.get("w_edge", 0.65)
    f_nms = fs.get("nms_radius", 10); f_topk = fs.get("top_k", 20)
    scales = fs.get("scales", cfg["scales"])
    cands, maps = _multiscale_search(ref_pre, srch_pre, scales, f_topk, f_nms, w_app, w_edg, [0.0])
    r_gray = ref_pre.get("gray_smooth", ref_pre["clahe"]).astype(np.float32)
    r_edge = ref_pre["sobel"]
    s_gray = srch_pre.get("gray_smooth", srch_pre["clahe"]).astype(np.float32)
    s_edge = srch_pre["sobel"]
    rh, rw = r_gray.shape
    for scale in scales:
        nh = max(8,int(rh*scale)); nw = max(8,int(rw*scale))
        if nh >= sh or nw >= sw: continue
        rgs = cv2.resize(r_gray,(nw,nh)); res = cv2.resize(r_edge,(nw,nh))
        comb, amap, emap = _combined_response(s_gray, s_edge, rgs, res, w_app, w_edg)
        if comb.size == 0: continue
        py,px = np.unravel_index(np.argmax(comb), comb.shape)
        cx,cy = int(px+nw//2), int(py+nh//2)
        a = float(amap[py,px]) if py<amap.shape[0] and px<amap.shape[1] else 0.0
        e = float(emap[py,px]) if py<emap.shape[0] and px<emap.shape[1] else 0.0
        cands.append(Candidate(x=cx,y=cy,scale=scale,appearance_score=a,edge_score=e))
    return cands, maps


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOCALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def localize(ref_img: np.ndarray, search_img: np.ndarray, cfg: dict = None, arch: str = None) -> dict:
    t0  = time.perf_counter()
    cfg = cfg or _CFG
    if ref_img.ndim == 3:    ref_img    = cv2.cvtColor(ref_img,    cv2.COLOR_BGR2GRAY)
    if search_img.ndim == 3: search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    sh, sw = search_img.shape
    pc = cfg["preprocessing"]; wt = cfg["weights"]

    ref_pre  = _preprocess(ref_img,    pc)
    srch_pre = _preprocess(search_img, pc)

    arch_detected = arch
    if arch_detected is None:
        ad_cfg = cfg.get("arch_detect", {})
        if ad_cfg.get("enabled", False):
            arch_detected = _detect_arch(srch_pre.get("gray_smooth", srch_pre["clahe"]), ad_cfg)
        else:
            arch_detected = "UNKNOWN"

    is_finfet = (arch_detected == "FINFET")
    top_k = cfg["top_k"]; nms_r = cfg["nms_radius"]

    # Candidate generation
    if is_finfet:
        cands, maps = _build_finfet_cands(ref_pre, srch_pre, sh, sw, cfg)
    else:
        cands, maps = _multiscale_search(ref_pre, srch_pre, cfg["scales"], top_k, nms_r, wt["appearance"], wt["edge"], [0.0])
        r_gray = ref_pre.get("gray_smooth", ref_pre["clahe"]).astype(np.float32)
        s_gray = srch_pre.get("gray_smooth", srch_pre["clahe"]).astype(np.float32)
        r_edge = ref_pre["sobel"]; s_edge = srch_pre["sobel"]; rh,rw = r_gray.shape
        for scale in cfg["scales"]:
            nh=max(8,int(rh*scale)); nw=max(8,int(rw*scale))
            if nh>=sh or nw>=sw: continue
            rgs=cv2.resize(r_gray,(nw,nh)); res=cv2.resize(r_edge,(nw,nh))
            comb,amap,emap = _combined_response(s_gray,s_edge,rgs,res,wt["appearance"],wt["edge"])
            if comb.size == 0: continue
            py,px = np.unravel_index(np.argmax(comb), comb.shape)
            cands.append(Candidate(x=px+nw//2, y=py+nh//2, scale=scale,
                                   appearance_score=float(amap[py,px]) if py<amap.shape[0] and px<amap.shape[1] else 0.0,
                                   edge_score=float(emap[py,px]) if py<emap.shape[0] and px<emap.shape[1] else 0.0))
        best_conf = cands[0].appearance_score if cands else 0.0
        if best_conf < cfg.get("rotation_confidence_threshold", 0.0):
            r_cands, _ = _multiscale_search(ref_pre, srch_pre, [1.0], top_k, nms_r,
                                            wt["appearance"], wt["edge"], cfg.get("rotations_deg", [-3,-2,-1,0,1,2,3]))
            cands = cands + r_cands

    if not cands:
        elapsed = (time.perf_counter()-t0)*1000
        return dict(x=sw//2, y=sh//2, confidence=0.0, elapsed_ms=elapsed,
                    candidates=[], arch_detected=arch_detected)

    eff_nms = cfg.get("finfet_search",{}).get("nms_radius", nms_r) if is_finfet else nms_r
    max_k   = top_k*5 if is_finfet else top_k*3
    cands   = _dedup(cands, eff_nms, max_k)

    if is_finfet:
        fw = cfg.get("finfet_weights", wt)
        pc_expand = cfg.get("finfet_search",{}).get("pc_expand_factor", 2.0)

        # FINAL COMPETITION PATH:
        # Always use classical Phase Correlation.
        # Siamese CNN is research-only and must not affect inference.
        cands = _score_phase_correlation(
            cands,
            ref_pre,
            srch_pre,
            pc_expand
        )

        cands = _score_periodicity(
            cands,
            ref_pre,
            srch_pre,
            cfg
        )

        cands = _rank(
            cands,
            fw,
            preferred_scale=1.0
        )
    else:
        margin_th = cfg.get("ncc_margin_threshold", 0.015)
        w_app, w_edg = wt["appearance"], wt["edge"]
        margin = (w_app*cands[0].appearance_score + w_edg*cands[0].edge_score) - \
                 (w_app*cands[1].appearance_score + w_edg*cands[1].edge_score) if len(cands) >= 2 else 1.0
        if margin >= margin_th:
            cands[0].neighborhood_score = 0.5; cands[0].periodicity_score = 0.5
            cands[0].final_score = w_app*cands[0].appearance_score + w_edg*cands[0].edge_score
        else:
            expand = cfg.get("periodicity",{}).get("expand_factor", 3.0)
            cands = _score_with_context(cands, ref_pre, srch_pre, expand)
            cands = _score_periodicity(cands, ref_pre, srch_pre, cfg)
            cands = _rank(cands, wt, preferred_scale=1.0)

    best = cands[0]
    ref_cfg = cfg["refinement"]
    fx, fy, fscore = _refine(best, ref_pre, srch_pre, ref_cfg)
    fx = int(np.clip(fx, 0, sw-1)); fy = int(np.clip(fy, 0, sh-1))
    elapsed = (time.perf_counter()-t0)*1000
    return dict(x=fx, y=fy, confidence=float(np.clip(fscore,0,1)),
                elapsed_ms=elapsed, candidates=cands, arch_detected=arch_detected)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DriftSense-AI: Wafer Inspection Localization")
    parser.add_argument("reference", help="Path to reference image")
    parser.add_argument("search",    help="Path to search image")
    parser.add_argument("--json",    action="store_true", help="Output JSON instead of 'x y'")
    parser.add_argument("--verbose", action="store_true", help="Print debug info to stderr")
    parser.add_argument("--arch",    choices=["DRAM","FINFET"], default=None,
                        help="Override architecture detection")
    args = parser.parse_args()

    def load_img(path):
        if not os.path.exists(path):
            print(f"ERROR: Image not found: {path}", file=sys.stderr); sys.exit(1)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"ERROR: Could not decode: {path}", file=sys.stderr); sys.exit(1)
        if img.ndim == 3 and img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.uint8)

    ref_img    = load_img(args.reference)
    search_img = load_img(args.search)

    if args.verbose:
        print(f"[INFO] Reference : {ref_img.shape}  {args.reference}", file=sys.stderr)
        print(f"[INFO] Search    : {search_img.shape}  {args.search}",  file=sys.stderr)

    result = localize(ref_img, search_img, arch=args.arch)

    if args.verbose:
        print(f"[INFO] Arch      : {result['arch_detected']}", file=sys.stderr)
        print("[INFO] Siamese   : disabled in competition path (using Phase Correlation)", file=sys.stderr)
        print(f"[INFO] Predicted : ({result['x']}, {result['y']})  conf={result['confidence']:.4f}  {result['elapsed_ms']:.1f}ms", file=sys.stderr)

    if args.json:
        out = {
            "x": result["x"],
            "y": result["y"],
            "confidence": round(result["confidence"], 4),
            "elapsed_ms": round(result.get("elapsed_ms", 0.0), 1),
            "arch_detected": result.get("arch_detected", "UNKNOWN"),
            "candidates_evaluated": len(result.get("candidates", []))
        }
        print(json.dumps(out))
    else:
        print(f"{result['x']} {result['y']}")


if __name__ == "__main__":
    main()
