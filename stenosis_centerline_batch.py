# Diameter Stenosis part from Sangram Sahoo
# stenosis_centerline_batch.py
# Coronary vessel centerline + diameter + %DS on a single image or a whole folder.

import os
import sys
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from blockage_pipeline import blockage_pipeline
from gamma_blackhat import gamma_blackhat_binarize_bgr


import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt

import networkx as nx

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, binary_fill_holes, gaussian_filter1d

from skimage import io, img_as_float
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import (
    closing,
    opening,
    remove_small_holes,
    remove_small_objects,
    disk,
    medial_axis,
    dilation,
)
from skimage.segmentation import clear_border


# ---------------------------
# Tunable parameters
# ---------------------------
ROLL_WINDOW = 80          # samples for proximal reference
ROLL_PERCENT = 0.90       # P90
EDGE_EXCLUDE = 15         # ignore ends when picking min diameter
MIN_REF_DIAM = 1.0        # px; ignore minima with tiny reference

# ---------------------------
# Core functions
# ---------------------------


def load_binary_mask(path: str) -> np.ndarray:
    """
    Load a pre-binarized mask image.
    Assumes dark lines = vessels, bright = background.
    Returns boolean mask with True for lumen.
    """
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {path}")

    thr = 0 if m.max() <= 1 else 127
    mask = (m < thr)  # invert: black = vessel, white = background

    mask = remove_small_objects(mask.astype(bool), min_size=64)
    return mask.astype(bool)


def lesion_region(diam, ref, pds, center_idx, pds_thresh=50.0, max_expand=40):
    """
    Starting from center_idx, expand left/right while %DS >= pds_thresh.
    Returns (i0, i1) inclusive indices of the lesion region.
    """
    n = len(diam)
    i0 = center_idx
    i1 = center_idx
    # expand left
    for k in range(center_idx - 1, max(-1, center_idx - max_expand) - 1, -1):
        if pds[k] >= pds_thresh:
            i0 = k
        else:
            break
    # expand right
    for k in range(center_idx + 1, min(n, center_idx + max_expand)):
        if pds[k] >= pds_thresh:
            i1 = k
        else:
            break
    return i0, i1


def enhance_gray(gray, clahe_clip=2.0, clahe_tile=(8, 8), sigma=0.8):
    g8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    g8 = clahe.apply(g8)
    g = g8.astype(np.float32) / 255.0
    g = gaussian_filter(g, sigma=sigma)
    return g


def build_fov_mask(gray, right_strip_px=110, bottom_strip_px=15):
    m = np.ones_like(gray, dtype=bool)
    m[:6, :] = False
    m[-6:, :] = False
    m[:, :6] = False
    m[:, -6:] = False
    if right_strip_px > 0:
        m[:, -right_strip_px:] = False
    if bottom_strip_px > 0:
        m[-bottom_strip_px:, :] = False
    return m


def segment_lumen_isolated_fixed(gray,
                                 fov_mask,
                                 min_obj=200,
                                 close_r=3,
                                 open_r=1,
                                 ridge_dilate=3):
    """
    Returns (lumen_mask, vesselness) suitable for medial-axis diameter.
    """
    g = enhance_gray(gray)  # [0..1]

    # black-hat (remove low-frequency background)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closing_g = cv2.morphologyEx((g * 255).astype(np.uint8),
                                 cv2.MORPH_CLOSE, se)
    blackhat = (closing_g.astype(np.float32) - (g * 255)).clip(0, 255) / 255.0

    # vesselness on inverted gray (dark vessels become bright)
    v = frangi(1.0 - g, beta=0.5, gamma=20)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)

    # candidate lumen by Otsu on blackhat within FOV
    bh = blackhat.copy()
    bh[~fov_mask] = 0
    t_bh = threshold_otsu(bh[fov_mask])
    lum0 = bh > t_bh

    # gate with vesselness ridges (dilated) so we keep only vascular structures
    ridge = v > np.percentile(v[fov_mask], 70)
    ridge = dilation(ridge, disk(ridge_dilate))
    mask = lum0 & ridge

    # clean-up
    mask = binary_fill_holes(mask)
    mask = closing(mask, disk(close_r))
    mask = opening(mask, disk(open_r))
    mask = remove_small_objects(mask, min_size=min_obj)
    mask = remove_small_holes(mask, area_threshold=min_obj // 2)
    mask = clear_border(mask)

    # keep largest component
    num_labels, lbl = cv2.connectedComponents(mask.astype(np.uint8),
                                              connectivity=8)
    if num_labels > 1:
        counts = np.bincount(lbl.ravel())
        keep = np.argmax(counts[1:]) + 1 if counts.size > 1 else 0
        mask = (lbl == keep)
    else:
        mask = (lbl > 0)

    return mask, v


def skeleton_and_distance(mask):
    skel, dist = medial_axis(mask, return_distance=True)
    y, x = np.nonzero(skel)
    coords = np.column_stack([y, x])
    diam = 2.0 * dist[skel]
    return skel, dist, coords, diam


def lumen_edges_morph(mask: np.ndarray) -> np.ndarray:
    """
    Simple morphological edge of the lumen mask.
    mask: bool array, True inside lumen.
    returns: bool array, True at lumen boundary.
    """
    m = mask.astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8))
    edge = (m & (~er))  # mask minus eroded interior
    return edge.astype(bool)


def unit_normal(P, i, ksize=5):
    """
    P: N x 2 array of [y,x] points (ordered centerline)
    i: index along centerline
    returns: (t, n) where n is the unit normal [ny, nx]
    """
    i0 = max(0, i - ksize)
    i1 = min(len(P) - 1, i + ksize)
    if i1 == i0:
        return np.array([0.0, 1.0], np.float32), np.array([1.0, 0.0], np.float32)

    dy = P[i1, 0] - P[i0, 0]
    dx = P[i1, 1] - P[i0, 1]
    t = np.array([dy, dx], np.float32)
    t /= (np.linalg.norm(t) + 1e-6)
    n = np.array([-t[1], t[0]], np.float32)
    return t, n


def diameters_from_edges(edge_map: np.ndarray,
                         centerline: np.ndarray,
                         max_search=30) -> np.ndarray:
    """
    Measure lumen diameter along the centerline using edge pixels.

    edge_map: bool array (True at lumen edges)
    centerline: N x 2 array of [y,x] ints (ordered skeleton)
    max_search: max pixels to search in each direction along the normal
    returns: N array of diameters in pixels
    """
    h, w = edge_map.shape
    P = centerline.astype(np.float32)
    diam = np.zeros(len(P), dtype=np.float32)

    for i in range(len(P)):
        cy, cx = P[i]
        _, n = unit_normal(P, i)

        pos_hit = None
        neg_hit = None

        # positive normal side
        for s in range(1, max_search + 1):
            py = int(round(cy + n[0] * s))
            px = int(round(cx + n[1] * s))
            if py < 0 or py >= h or px < 0 or px >= w:
                break
            if edge_map[py, px]:
                pos_hit = (py, px)
                break

        # negative normal side
        for s in range(1, max_search + 1):
            ny = int(round(cy - n[0] * s))
            nx_ = int(round(cx - n[1] * s))
            if ny < 0 or ny >= h or nx_ < 0 or nx_ >= w:
                break
            if edge_map[ny, nx_]:
                neg_hit = (ny, nx_)
                break

        if pos_hit is not None and neg_hit is not None:
            y1, x1 = pos_hit
            y0, x0 = neg_hit
            diam[i] = np.hypot(y1 - y0, x1 - x0)
        else:
            diam[i] = 0.0

    return diam


def build_graph_from_skeleton(skel):
    ys, xs = np.nonzero(skel)
    idx = {(y, x): i for i, (y, x) in enumerate(zip(ys, xs))}
    G = nx.Graph()
    for i, (y, x) in enumerate(zip(ys, xs)):
        G.add_node(i, y=y, x=x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in idx:
                    j = idx[(ny, nx_)]
                    if not G.has_edge(i, j):
                        w = 1.4142 if (dy != 0 and dx != 0) else 1.0
                        G.add_edge(i, j, weight=w)
    return G


def _farthest_node_weighted(G, source, weight="weight"):
    """Return the farthest node and its distance from `source` using Dijkstra."""
    lengths = nx.single_source_dijkstra_path_length(G, source, weight=weight)
    far_node = max(lengths, key=lengths.get)
    return far_node, lengths[far_node]


def longest_path_on_lcc(G):
    """Approximate diameter path of the largest connected component via two Dijkstra sweeps."""
    if G.number_of_nodes() == 0:
        return [], None
    lcc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc_nodes).copy()

    # pick an arbitrary node to start
    start = next(iter(H.nodes))
    # first sweep
    far1, _ = _farthest_node_weighted(H, start, weight="weight")
    # second sweep
    far2, _ = _farthest_node_weighted(H, far1, weight="weight")

    path = nx.shortest_path(H, far1, far2, weight="weight")
    return path, H


def rolling_ref(diam, window=ROLL_WINDOW, perc=0.9):
    s = pd.Series(diam)
    ref = s.rolling(window,
                    min_periods=max(10, window // 4)).quantile(perc)
    ref = ref.bfill().ffill().to_numpy()
    return ref


def intensity_profile_along_path(img, yx_ord, smooth_sigma=1.5):
    """
    Sample image intensity along the ordered centerline and optionally smooth it.
    img: float grayscale [0..1]
    yx_ord: N x 2 array of (row, col) along centerline
    """
    ys = np.clip(yx_ord[:, 0].astype(int), 0, img.shape[0] - 1)
    xs = np.clip(yx_ord[:, 1].astype(int), 0, img.shape[1] - 1)
    I = img[ys, xs].astype(np.float32)
    if smooth_sigma is not None and smooth_sigma > 0:
        I = gaussian_filter1d(I, sigma=smooth_sigma)
    return I


def intensity_ref_and_drop(I_ord, window=ROLL_WINDOW, perc=0.9):
    """
    Build a 'healthy' reference intensity profile and compute % intensity drop.
    We assume lumen is normally bright; stenosis → darker segment (drop in intensity).
    """
    ref_I = rolling_ref(I_ord, window=window, perc=perc)
    pid = (1.0 - I_ord / (ref_I + 1e-8)) * 100.0
    return ref_I, pid


def measure_stenosis_on_profile(yx_ord, d_ord):
    """
    yx_ord: N x 2 ordered centerline coords
    d_ord:  N     ordered diameters (e.g., edge-based)
    returns: (yx_ord, d_ord, ref_ord, pds, min_idx)
    """
    ref_ord = rolling_ref(d_ord, window=ROLL_WINDOW, perc=ROLL_PERCENT)
    pds = (1.0 - (d_ord / (ref_ord + 1e-8))) * 100.0

    lo = EDGE_EXCLUDE
    hi = len(d_ord) - EDGE_EXCLUDE
    idx = np.argmin(d_ord[lo:hi]) + lo if hi > lo else np.argmin(d_ord)
    if ref_ord[idx] < MIN_REF_DIAM:
        for k in np.argsort(d_ord):
            if ref_ord[k] >= MIN_REF_DIAM:
                idx = int(k)
                break
    return yx_ord, d_ord, ref_ord, pds, int(idx)


def draw_overlay_targeted(base_gray,
                          yx_ord,
                          d_ord,
                          ref_ord,
                          pds,
                          idx_center,
                          idx_prox,
                          idx_dist,
                          out_path):
    """
    Draw a precise marker:
      - red cross at idx_center
      - proximal/distal edge ticks at idx_prox/idx_dist
      - a short caliper (perpendicular segment) with length = local diameter
    """

    base8 = cv2.normalize(base_gray, None, 0, 255,
                          cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR)

    def _unit_normal(P, i, ksize=10):
        i0 = max(0, i - ksize)
        i1 = min(len(P) - 1, i + ksize)
        if i1 == i0:
            return np.array([0.0, 1.0]), np.array([1.0, 0.0])
        dy = P[i1, 0] - P[i0, 0]
        dx = P[i1, 1] - P[i0, 1]
        t = np.array([dy, dx], np.float32)
        t /= (np.linalg.norm(t) + 1e-6)
        n = np.array([-t[1], t[0]], np.float32)
        return t, n

    P = yx_ord.astype(np.int32)  # (row, col)

    # ---- Center marker (tightest point) ----
    cy, cx = int(P[idx_center, 0]), int(P[idx_center, 1])
    diam_px = float(d_ord[idx_center])

    # cross
    cv2.drawMarker(rgb, (cx, cy), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS,
                   markerSize=14,
                   thickness=2)

    # perpendicular caliper (just a short segment equal to diameter)
    _, n = _unit_normal(P, idx_center)
    v = (n * (diam_px / 2.0)).astype(np.int32)
    p1 = (int(cx - v[1]), int(cy - v[0]))
    p2 = (int(cx + v[1]), int(cy + v[0]))
    cv2.line(rgb, p1, p2, (0, 0, 255), 2)

    # ---- Proximal / distal edge ticks ----
    for k in [idx_prox, idx_dist]:
        ky, kx = int(P[k, 0]), int(P[k, 1])
        _, n2 = _unit_normal(P, k)
        v2 = (n2 * 3).astype(np.int32)  # 6 px tick
        q1 = (int(kx - v2[1]), int(ky - v2[0]))
        q2 = (int(kx + v2[1]), int(ky + v2[0]))
        cv2.line(rgb, q1, q2, (0, 0, 255), 2)

    # label: %DS at center
    pds_val = float(pds[idx_center])
    cv2.putText(rgb, f"{pds_val:.0f}% DS", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, rgb)


def draw_overlay_severe(base_gray,
                        yx_ord,
                        d_ord,
                        ref_ord,
                        pds,
                        severe_idxs,
                        out_path):
    """
    Draw skeleton + ALL stenosis points with %DS >= threshold (e.g. 80%).
    Each severe index gets:
      - red cross
      - short perpendicular caliper
      - %DS label
    """

    base8 = cv2.normalize(base_gray, None, 0, 255,
                          cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR)

    # skeleton in white
    skel = np.zeros_like(base8, dtype=bool)
    sy, sx = yx_ord[:, 0].astype(int), yx_ord[:, 1].astype(int)
    skel[sy, sx] = True
    yy, xx = np.nonzero(skel)
    rgb[yy, xx] = (255, 255, 255)

    def _unit_normal(P, i, ksize=10):
        i0 = max(0, i - ksize)
        i1 = min(len(P) - 1, i + ksize)
        if i1 == i0:
            return np.array([0.0, 1.0]), np.array([1.0, 0.0])
        dy = P[i1, 0] - P[i0, 0]
        dx = P[i1, 1] - P[i0, 1]
        t = np.array([dy, dx], np.float32)
        t /= (np.linalg.norm(t) + 1e-6)
        n = np.array([-t[1], t[0]], np.float32)
        return t, n

    P = yx_ord.astype(np.int32)

    for idx in severe_idxs:
        cy, cx = int(P[idx, 0]), int(P[idx, 1])
        diam_px = float(d_ord[idx])

        # cross
        cv2.drawMarker(rgb, (cx, cy), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS,
                       markerSize=12,
                       thickness=2)

        # short perpendicular caliper
        _, n = _unit_normal(P, idx)
        v = (n * (diam_px / 2.0)).astype(np.int32)
        p1 = (int(cx - v[1]), int(cy - v[0]))
        p2 = (int(cx + v[1]), int(cy + v[0]))
        cv2.line(rgb, p1, p2, (0, 0, 255), 2)

        # label %DS
        cv2.putText(rgb, f"{pds[idx]:.0f}%", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, rgb)


def pick_stenoses(diam, pds, rel_drop=0.4, abs_drop=2.0, window=15):
    """
    Detect sudden focal diameter drops (stenoses).
    rel_drop: fractional diameter drop (e.g. 0.4 = 40%)
    abs_drop: absolute drop in pixels required
    window: neighborhood size (samples)
    """
    d = np.asarray(diam)
    stenoses = []

    grad = np.gradient(d)
    mins, _ = find_peaks(-d)  # local minima

    for i in mins:
        left = max(0, i - window)
        right = min(len(d), i + window)
        d_left = np.max(d[left:i]) if i > left else d[i]
        d_right = np.max(d[i:right]) if i < right else d[i]
        drop = max(d_left - d[i], d_right - d[i])
        rel = drop / (max(d_left, d_right) + 1e-8)

        if (drop >= abs_drop or rel >= rel_drop) and grad[i - 1] < 0 and grad[i + 1] > 0:
            stenoses.append(i)

    stenoses = sorted(stenoses, key=lambda j: d[j])
    return stenoses


def pick_stenoses_geo_int(diam, pds, pid,
                          rel_drop=0.4,
                          abs_drop=2.0,
                          int_drop=15.0,
                          window=15):
    """
    Combine geometry (diameter) + intensity.
    - First use geometric logic (`pick_stenoses`)
    - Then keep only those where the percent intensity drop (PID) is large enough.
    """
    base_cands = pick_stenoses(diam, pds,
                               rel_drop=rel_drop,
                               abs_drop=abs_drop,
                               window=window)
    base_cands = list(base_cands)
    if not base_cands:
        return []

    pid = np.asarray(pid)
    strong = [i for i in base_cands if pid[i] >= int_drop]
    return strong if strong else base_cands


# ---------------------------
# Per-image pipeline
# ---------------------------

def run_one_image(input_path, out_prefix, px=None,
                  pds_thresh=50.0, topk=3, suppression=20,
                  mark_deepest=False, mark_index=None,
                  debug_plot=False, mask_mode=False,
                  use_gamma_seg=False,
                  gamma_val=1.0, gamma_kernel=31, gamma_thresh=100):

    # 0) Load base grayscale (used for overlay background)
    img = img_as_float(io.imread(input_path, as_gray=True))

    intensity8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f"{out_prefix}_intensity.png", intensity8)

    heat = cv2.applyColorMap(intensity8, cv2.COLORMAP_JET)
    alpha = 0.45
    overlay_heat = cv2.addWeighted(
        heat, alpha,
        cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR), 1 - alpha, 0
    )
    cv2.imwrite(f"{out_prefix}_intensity_heatmap.png", overlay_heat)

    # 🔹 Run teammate's blockage detector on *this* image
    try:
        run_rajiv_blockage(input_path, out_prefix)
    except Exception as e:
        print(f"[WARN] teammate blockage_pipeline failed on {input_path}: {e}")


    # 1) Get lumen mask:
    #    a) --mask_mode: use external binary mask file
    #    b) --use_gamma_seg: gamma+blackhat+threshold binarization
    #    c) else: our default Frangi + black-hat segmentation
    if mask_mode:
        lumen_mask = load_binary_mask(input_path)
        vesselness = None

    elif use_gamma_seg:
        #  gamma + black-hat + threshold
        bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read image for gamma seg: {input_path}")

        bin_img = gamma_blackhat_binarize_bgr(
            bgr,
            gamma=gamma_val,
            kernel_size=gamma_kernel,
            thresh=gamma_thresh
        )

        #Shweta's code returns bright = vessel; we want a boolean mask
        lumen_mask = bin_img > 0

        # no 'vesselness' map here
        vesselness = None

    else:
        # default: our Frangi + black-hat pipeline
        fov_mask = build_fov_mask(img, right_strip_px=110, bottom_strip_px=15)
        lumen_mask, vesselness = segment_lumen_isolated_fixed(
            img, fov_mask,
            min_obj=200, close_r=3, open_r=1, ridge_dilate=3
        )

    # 2) Skeleton + distance (DT diameters are QC only now)
    skel, dist, coords, diam_dt = skeleton_and_distance(lumen_mask)

    # 3) Save intermediates
    if vesselness is not None:
        cv2.imwrite(f"{out_prefix}_vesselness.png",
                    (vesselness * 255).astype(np.uint8))
    cv2.imwrite(f"{out_prefix}_mask.png",
                (lumen_mask.astype(np.uint8)) * 255)

    base8 = cv2.normalize(img, None, 0, 255,
                          cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR)
    ys, xs = np.nonzero(skel)
    rgb[ys, xs] = (255, 255, 255)
    cv2.imwrite(f"{out_prefix}_skeleton.png", rgb)

    # 4) Order centerline along main path
    G = build_graph_from_skeleton(skel)
    path, H = longest_path_on_lcc(G)
    if not path:
        raise RuntimeError("Empty skeleton graph.")

    yx_to_idx = {(r, c): k for k, (r, c) in enumerate(map(tuple, coords))}
    path_idx = np.array(
        [yx_to_idx[(H.nodes[n]['y'], H.nodes[n]['x'])] for n in path]
    )

    yx_ord = coords[path_idx]

    # 4b) Edge map from lumen mask
    edges = lumen_edges_morph(lumen_mask)

    # 4c) Diameters from edges along the skeleton
    d_ord = diameters_from_edges(edges, yx_ord, max_search=30)

    # 5) Diameter profile, reference, %DS
    yx_ord, d_ord, ref_ord, pds, _ = measure_stenosis_on_profile(yx_ord,
                                                                 d_ord)
    center_idx = 0

    # 5a) Diameter debug plot
    if debug_plot:
        try:
            plt.figure(figsize=(8, 3))
            plt.plot(d_ord, label="Diameter (px)")
            plt.plot(ref_ord, "--", label="Ref diameter (P90)")
            plt.title(f"Diameter profile: {os.path.basename(input_path)}")
            plt.xlabel("Centerline index")
            plt.ylabel("Diameter (px)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_diam_profile.png", dpi=160)
            plt.close()
        except Exception as e:
            print(f"[WARN] diameter plot failed: {e}")

    # 5b) Intensity profile + % intensity drop
    I_ord = intensity_profile_along_path(img, yx_ord, smooth_sigma=1.5)
    I_ref, pid = intensity_ref_and_drop(I_ord,
                                        window=ROLL_WINDOW,
                                        perc=ROLL_PERCENT)

    # Intensity profile plot
    if debug_plot:
        try:
            plt.figure(figsize=(8, 3))
            plt.plot(I_ord, label="Intensity")
            plt.plot(I_ref, "--", label="Ref Intensity (P90)")
            plt.plot(pid, label="% Intensity Drop")
            plt.title(f"Intensity profile: {os.path.basename(input_path)}")
            plt.xlabel("Centerline index")
            plt.ylabel("Intensity / %Drop")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_intensity_profile.png", dpi=160)
            plt.close()
        except Exception as e:
            print(f"[WARN] intensity plot failed: {e}")

    # 6) Severe indices (%DS >= 80)
    # 6b) we still keep severe_idxs for statistics, but we don't use it for drawing
    severe_idxs = np.where(pds >= 80.0)[0]

    overlay_path = f"{out_prefix}_overlay.png"
    overlay_made = False

    if len(d_ord) == 0 or len(pds) == 0:
        print(f"[skip] {os.path.basename(input_path)}: empty diameter profile; no overlay.")
    elif pds[center_idx] < 20:
        print(f"[skip] {os.path.basename(input_path)}: %DS={pds[center_idx]:.1f} < 20; not drawing.")
    else:
        # find lesion extent around chosen index
        i0, i1 = lesion_region(
            d_ord, ref_ord, pds,
            center_idx,
            pds_thresh=float(pds_thresh),
            max_expand=40
        )
        # draw your clean overlay (cross + caliper)
        draw_overlay_targeted(
            img, yx_ord, d_ord, ref_ord, pds,
            center_idx, i0, i1, overlay_path
        )
        overlay_made = True

    # 7) Choose index(es) to mark
    if len(d_ord) == 0:
        idxs = []
        center_idx = 0
    else:
        if mark_index is not None:
            idxs = [int(np.clip(mark_index, 0, len(d_ord) - 1))]
        elif mark_deepest:
            idxs = [int(np.argmin(d_ord))]
        else:
            # geometry + intensity combined
            idxs = pick_stenoses_geo_int(
                d_ord, pds, pid,
                rel_drop=0.30,   # 30% relative diameter drop
                abs_drop=1.5,    # or >= 1.5 px absolute drop
                int_drop=15.0,   # and ~15% intensity drop
                window=12
            )
            if len(idxs) == 0:
                idxs = [int(np.argmin(d_ord))]

        center_idx = int(np.clip(idxs[0], 0, len(d_ord) - 1))


    # 8) Lesion extent & overlay
    i0, i1 = lesion_region(d_ord, ref_ord, pds, center_idx,
                           pds_thresh=float(pds_thresh),
                           max_expand=40)

    overlay_path = f"{out_prefix}_overlay.png"
    overlay_made = False

    if len(severe_idxs) > 0:
        draw_overlay_severe(img, yx_ord, d_ord, ref_ord, pds,
                            severe_idxs, overlay_path)
        overlay_made = True
    elif len(d_ord) > 0 and pds[center_idx] >= 20:
        draw_overlay_targeted(img, yx_ord, d_ord, ref_ord, pds,
                              center_idx, i0, i1, overlay_path)
        overlay_made = True
    else:
        print(f"[skip] {os.path.basename(input_path)}: "
              f"%DS={pds[center_idx]:.1f} < 20; not drawing.")

    # 9) Build per-image CSV
    curve_idx = np.arange(len(d_ord))
    df = pd.DataFrame({
        "curve_idx": curve_idx,
        "y": yx_ord[:, 0],
        "x": yx_ord[:, 1],
        "diameter_px": d_ord,
        "ref_diameter_px": ref_ord,
        "percent_diameter_stenosis": pds,
        "intensity": I_ord,
        "ref_intensity": I_ref,
        "percent_intensity_drop": pid
    })
    if px is not None:
        df["diameter_mm"] = df["diameter_px"] * px
        df["ref_diameter_mm"] = df["ref_diameter_px"] * px

    csv_path = f"{out_prefix}.csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)

    detections = [{
        "idx": int(i),
        "r": int(yx_ord[i, 0]),
        "c": int(yx_ord[i, 1]),
        "diam_px": float(d_ord[i]),
        "ref_px": float(ref_ord[i]),
        "pds": float(pds[i])
    } for i in idxs]
    det_csv = f"{out_prefix}_detections.csv"
    pd.DataFrame(detections).to_csv(det_csv, index=False)

    # 9b) Count %DS ranges (ignore curve edges to avoid border artifacts)
    valid_mask = (curve_idx >= EDGE_EXCLUDE) & \
                 (curve_idx < len(d_ord) - EDGE_EXCLUDE)
    if len(d_ord) > 0 and np.any(valid_mask):
        pds_valid = pds[valid_mask]
        mask_80_90 = (pds_valid >= 80.0) & (pds_valid < 90.0)
        mask_ge90 = (pds_valid >= 90.0)

        num_80_90 = int(np.count_nonzero(mask_80_90))
        num_ge90 = int(np.count_nonzero(mask_ge90))
        max_pds_valid = float(np.max(pds_valid))
    else:
        num_80_90 = 0
        num_ge90 = 0
        max_pds_valid = float("nan")

    # 10) Summary dict
    first = detections[0] if detections else {}
    summary = {
        "image": input_path,
        "overlay_path": overlay_path if overlay_made else "",
        "per_image_csv": csv_path,
        "detections_csv": det_csv if detections else "",
        "num_detections": len(detections),
        "first_x": int(first.get("c", -1)),
        "first_y": int(first.get("r", -1)),
        "first_diam_px": float(first.get("diam_px", np.nan)),
        "first_ref_px": float(first.get("ref_px", np.nan)),
        "first_%DS": float(first.get("pds", np.nan)),
        "num_points_80_90": num_80_90,
        "num_points_ge90": num_ge90,
        "max_%DS_valid": max_pds_valid
    }
    if px is not None and detections:
        summary.update({
            "first_diam_mm": float(first["diam_px"]) * px,
            "first_ref_mm": float(first["ref_px"]) * px
        })
    return summary


# ---------------------------
# Batch utilities
# ---------------------------

VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_images(root, pattern="*.png", recursive=False):
    if os.path.isfile(root):
        return [root]
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} not found.")

    if recursive:
        paths = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    else:
        paths = glob.glob(os.path.join(root, pattern))

    if pattern == "*.png":
        extra = []
        for ext in VALID_EXT:
            extra += glob.glob(
                os.path.join(root, "**" if recursive else "", f"*{ext}"),
                recursive=recursive
            )
        paths = sorted(set(paths + extra))

    return [p for p in paths
            if os.path.splitext(p)[1].lower() in VALID_EXT]


def safe_stem(path):
    s = os.path.splitext(os.path.basename(path))[0]
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def run_rajiv_blockage(image_path: str, out_prefix: str):
    """
    image_path : full path to the input (e.g. D:\\Work\\CV\\Project\\Data\\14.png)
    out_prefix: output prefix; we only use its folder as save_folder.
    """
    save_folder = os.path.dirname(out_prefix) or "."
    blockage_pipeline(image_path, save_folder)


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Centerline + diameter + stenosis on images or a folder"
    )
    ap.add_argument("--in", dest="inp", required=True,
                    help="input image or directory")
    ap.add_argument("--out", dest="out", required=True,
                    help="output dir OR file prefix for single image")
    ap.add_argument("--px", dest="px", type=float, default=None,
                    help="mm per pixel (optional)")
    ap.add_argument("--pattern", default="*.png",
                    help="glob for folder mode (default: *.png)")
    ap.add_argument("--recursive", action="store_true",
                    help="recurse into subfolders")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers for folder mode")
    ap.add_argument("--pds_thresh", type=float, default=50.0,
                    help="Flag as stenosis if %%DS >= this (default 50)")
    ap.add_argument("--topk", type=int, default=3,
                    help="Max number of stenoses to report per image")
    ap.add_argument("--suppression", type=int, default=20,
                    help="Non-maximum suppression window on the profile")
    ap.add_argument("--mark_deepest", action="store_true",
                    help="Force marking the global minimum diameter")
    ap.add_argument("--mark_index", type=int, default=None,
                    help="Mark a specific centerline index from plot")
    ap.add_argument("--debug_plot", action="store_true",
                    help="Save diameter/intensity profile plots")
    ap.add_argument("--mask_mode", action="store_true",
                    help="Input is already a binarized vessel mask."
                         " Skip segmentation.")
    ap.add_argument("--use_gamma_seg", action="store_true",
                    help="gamma+blackhat+threshold binarization for lumen mask")
    ap.add_argument("--gamma_val", type=float, default=1.0,
                    help="Gamma for binarization (default 1.0)")
    ap.add_argument("--gamma_kernel", type=int, default=31,
                    help="Black-hat kernel size for binarization (default 31)")
    ap.add_argument("--gamma_thresh", type=int, default=100,
                    help="Threshold for binarization (0–255, default 100)")
    
    return ap.parse_args()


def main():
    args = parse_args()
    inp = args.inp
    out = args.out
    px = args.px

    if os.path.isfile(inp):
        # single image mode
        if os.path.isdir(out):
            out_prefix = os.path.join(out, safe_stem(inp))
        else:
            out_prefix = out

        res = run_one_image(
            inp, out_prefix, px=px,
            pds_thresh=args.pds_thresh,
            topk=args.topk,
            suppression=args.suppression,
            mark_deepest=args.mark_deepest,
            mark_index=args.mark_index,
            debug_plot=args.debug_plot,
            mask_mode=args.mask_mode,
            use_gamma_seg=args.use_gamma_seg,
            gamma_val=args.gamma_val,
            gamma_kernel=args.gamma_kernel,
            gamma_thresh=args.gamma_thresh,
        )
        print(pd.Series(res).to_string())
        return

    # folder mode
    images = list_images(inp, pattern=args.pattern, recursive=args.recursive)
    if not images:
        print("No images found. Check --pattern/--recursive.",
              file=sys.stderr)
        sys.exit(1)
    os.makedirs(out, exist_ok=True)

    summaries = []
    errors = []

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut2img = {}
            for im in images:
                pref = os.path.join(out, safe_stem(im))
                fut = ex.submit(
                    run_one_image, im, pref, px,
                    args.pds_thresh,
                    args.topk,
                    args.suppression,
                    args.mark_deepest,
                    args.mark_index,
                    args.debug_plot,
                    args.mask_mode
                )
                fut2img[fut] = im
            for fut in as_completed(fut2img):
                im = fut2img[fut]
                try:
                    summaries.append(fut.result())
                    print(f"[OK] {im}")
                except Exception as e:
                    errors.append((im, str(e)))
                    print(f"[ERR] {im}: {e}", file=sys.stderr)
    else:
        for im in images:
            try:
                pref = os.path.join(out, safe_stem(im))
                summaries.append(
                    run_one_image(
                        im, pref, px,
                        args.pds_thresh,
                        args.topk,
                        args.suppression,
                        args.mark_deepest,
                        args.mark_index,
                        args.debug_plot,
                        args.mask_mode,
                        args.use_gamma_seg, args.gamma_val,
                        args.gamma_kernel, args.gamma_thresh
                    )
                )
                print(f"[OK] {im}")
            except Exception as e:
                errors.append((im, str(e)))
                print(f"[ERR] {im}: {e}", file=sys.stderr)

    # write master summary
    if summaries:
        df = pd.DataFrame(summaries)
        summary_path = os.path.join(out, "summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\nSaved master summary: {summary_path}")
        if px is not None:
            print("Units: diameters also reported in mm using --px.")

        # Extra: only images with at least one point in [80, 90)% DS
        if "num_points_80_90" in df.columns:
            df_80_90 = df[df["num_points_80_90"] > 0].copy()
            if not df_80_90.empty:
                path_80_90 = os.path.join(out, "summary_80_90.csv")
                df_80_90.to_csv(path_80_90, index=False)
                print(f"Saved 80–90% DS summary: {path_80_90}")

    if errors:
        print("\nSome images failed:", file=sys.stderr)
        for im, msg in errors:
            print(f"- {im}: {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
