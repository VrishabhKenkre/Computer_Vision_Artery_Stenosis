# stenosis_centerline_batch.py
# Coronary vessel centerline + diameter + %DS on a single image or a whole folder.

import os, sys, argparse, glob, traceback
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import io, img_as_float
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import (closing, opening, remove_small_holes,
                                remove_small_objects, disk, medial_axis)
import networkx as nx
from skimage.morphology import remove_small_objects, remove_small_holes, closing, opening, disk
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu, frangi
from scipy.ndimage import gaussian_filter

from skimage.filters import threshold_sauvola, frangi, threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, remove_small_holes, closing, opening, dilation, disk
from scipy.ndimage import gaussian_filter, binary_fill_holes


from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import (closing, opening, remove_small_holes, remove_small_objects,
                                disk, dilation)
from skimage.segmentation import clear_border

# ---------------------------
# Tunable parameters
# ---------------------------
MIN_OBJECT_SIZE = 200
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAUSS_SIGMA = 1.0
FRANGI_BETA = 0.5
FRANGI_GAMMA = 25
CLOSE_RADIUS = 3
OPEN_RADIUS  = 1

ROLL_WINDOW   = 80         # samples for proximal reference
ROLL_PERCENT  = 0.90       # P90
EDGE_EXCLUDE  = 15         # ignore ends when picking min diameter
MIN_REF_DIAM  = 1.0        # px; ignore minima with tiny reference

# ---------------------------
# Core functions
# ---------------------------

def lesion_region(diam, ref, pds, center_idx, pds_thresh=50.0, max_expand=40):
    """
    Starting from center_idx, expand left/right while %DS >= pds_thresh.
    Returns (i0, i1) inclusive indices of the lesion region.
    """
    n = len(diam)
    i0 = center_idx
    i1 = center_idx
    # expand left
    for k in range(center_idx-1, max(-1, center_idx-max_expand)-1, -1):
        if pds[k] >= pds_thresh:
            i0 = k
        else:
            break
    # expand right
    for k in range(center_idx+1, min(n, center_idx+max_expand)):
        if pds[k] >= pds_thresh:
            i1 = k
        else:
            break
    return i0, i1



def enhance_gray(gray, clahe_clip=2.0, clahe_tile=(8,8), sigma=0.8):
    g8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    g8 = clahe.apply(g8)
    g = g8.astype(np.float32)/255.0
    g = gaussian_filter(g, sigma=sigma)
    return g

def build_fov_mask(gray, right_strip_px=110, bottom_strip_px=15):
    m = np.ones_like(gray, dtype=bool)
    m[:6,:]=False; m[-6:,:]=False; m[:,:6]=False; m[:,-6:]=False
    if right_strip_px>0:  m[:, -right_strip_px:] = False
    if bottom_strip_px>0: m[-bottom_strip_px:, :] = False
    return m

def segment_lumen_isolated_fixed(gray, fov_mask,
                                 min_obj=200, close_r=3, open_r=1, ridge_dilate=3):
    """
    Returns (lumen_mask, vesselness) suitable for medial-axis diameter.
    """
    g = enhance_gray(gray)                     # [0..1]
    inv = 1.0 - g                              # arteries darker → higher in inv
    # black-hat (remove low-frequency background)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    closing_g = cv2.morphologyEx((g*255).astype(np.uint8), cv2.MORPH_CLOSE, se)
    blackhat = (closing_g.astype(np.float32) - (g*255)).clip(0,255) / 255.0

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
    mask = remove_small_holes(mask, area_threshold=min_obj//2)
    mask = clear_border(mask)

    # keep largest component
    num_labels, lbl = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        counts = np.bincount(lbl.ravel())
        keep = np.argmax(counts[1:]) + 1 if counts.size>1 else 0
        mask = (lbl == keep)
    else:
        mask = (lbl > 0)

    return mask, v

def enhance_vessels(gray):
    g8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    g8 = clahe.apply(g8)
    g = g8.astype(np.float32) / 255.0
    g = gaussian_filter(g, sigma=GAUSS_SIGMA)
    v = frangi(1.0 - g, beta=FRANGI_BETA, gamma=FRANGI_GAMMA)  # invert: vessels dark
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return v

def segment_vessels(v):
    t = threshold_otsu(v)
    m = v > t
    m = binary_fill_holes(m)
    m = remove_small_objects(m, MIN_OBJECT_SIZE)
    m = remove_small_holes(m, area_threshold=MIN_OBJECT_SIZE // 2)
    m = closing(m, disk(CLOSE_RADIUS))
    m = opening(m, disk(OPEN_RADIUS))
    return m

def skeleton_and_distance(mask):
    skel, dist = medial_axis(mask, return_distance=True)
    y, x = np.nonzero(skel)
    coords = np.column_stack([y, x])
    diam = 2.0 * dist[skel]
    return skel, dist, coords, diam

def build_graph_from_skeleton(skel):
    ys, xs = np.nonzero(skel)
    idx = { (y,x): i for i, (y,x) in enumerate(zip(ys, xs)) }
    G = nx.Graph()
    for i, (y, x) in enumerate(zip(ys, xs)):
        G.add_node(i, y=y, x=x)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y+dy, x+dx
                if (ny, nx_) in idx:
                    j = idx[(ny, nx_)]
                    if not G.has_edge(i, j):
                        w = 1.4142 if (dy!=0 and dx!=0) else 1.0
                        G.add_edge(i, j, weight=w)
    return G

def _farthest_node_weighted(G, source, weight="weight"):
    """Return the farthest node and its distance from `source` using Dijkstra."""
    # lengths is a dict: node -> distance
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


def rolling_ref(diam, window=80, perc=0.9):
    s = pd.Series(diam)
    ref = s.rolling(window, min_periods=max(10, window//4)).quantile(perc)
    ref = ref.bfill().ffill().to_numpy()   # <- instead of fillna(method="bfill").fillna(method="ffill")
    return ref


def measure_stenosis_along_path(coords, diam, path_order):
    yx = coords[path_order, :]
    d  = diam[path_order]
    ref = rolling_ref(d, window=ROLL_WINDOW, perc=ROLL_PERCENT)
    pds = (1.0 - (d / (ref + 1e-8))) * 100.0
    lo = EDGE_EXCLUDE
    hi = len(d) - EDGE_EXCLUDE
    idx = np.argmin(d[lo:hi]) + lo if hi > lo else np.argmin(d)
    if ref[idx] < MIN_REF_DIAM:
        for k in np.argsort(d):
            if ref[k] >= MIN_REF_DIAM:
                idx = int(k); break
    return yx, d, ref, pds, int(idx)

def draw_overlay_targeted(base_gray, yx_ord, d_ord, ref_ord, pds, idx_center, idx_prox, idx_dist, out_path):
    """
    Draw a precise marker:
      - red cross at idx_center
      - proximal/distal edge ticks at idx_prox/idx_dist
      - a short caliper (perpendicular segment) with length = local diameter
    """

    base8 = cv2.normalize(base_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR)

    # --- Helper to compute local tangent & normal from ordered skeleton ---
    def unit_normal(P, i, ksize=10):
        i0 = max(0, i-ksize)
        i1 = min(len(P)-1, i+ksize)
        if i1 == i0:
            return np.array([0.0,1.0]), np.array([1.0,0.0])
        dy = P[i1,0] - P[i0,0]
        dx = P[i1,1] - P[i0,1]
        t = np.array([dy, dx], np.float32)
        t /= (np.linalg.norm(t) + 1e-6)
        n = np.array([-t[1], t[0]], np.float32)
        return t, n


    P = yx_ord.astype(np.int32)  # (row, col)
    H, W = rgb.shape[:2]

    # ---- Center marker (tightest point) ----
    cy, cx = int(P[idx_center,0]), int(P[idx_center,1])
    diam_px = float(d_ord[idx_center])
    rad = max(1, int(round(diam_px/2.0)))

    # cross
    cv2.drawMarker(rgb, (cx, cy), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

    # perpendicular caliper (just a short segment equal to diameter)
    _, n = unit_normal(P, idx_center)
    v = (n * (diam_px/2.0)).astype(np.int32)
    p1 = (int(cx - v[1]), int(cy - v[0]))
    p2 = (int(cx + v[1]), int(cy + v[0]))
    cv2.line(rgb, p1, p2, (0,0,255), 2)

    # ---- Proximal / distal edge ticks (small 6-px ticks perpendicular to centerline) ----
    for k in [idx_prox, idx_dist]:
        ky, kx = int(P[k,0]), int(P[k,1])
        _, n2 = unit_normal(P, k)
        v2 = (n2 * 3).astype(np.int32)  # 6 px long tick
        q1 = (int(kx - v2[1]), int(ky - v2[0]))
        q2 = (int(kx + v2[1]), int(ky + v2[0]))
        cv2.line(rgb, q1, q2, (0,0,255), 2)

    # optional label: %DS at center
    pds_val = float(pds[idx_center])
    cv2.putText(rgb, f"{pds_val:.0f}% DS", (cx+8, cy-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, rgb)



def run_one_image(input_path, out_prefix, px=None,
                  pds_thresh=50.0, topk=3, suppression=20,
                  mark_deepest=False, mark_index=None, debug_plot=False):
    img = img_as_float(io.imread(input_path, as_gray=True))
    fov_mask = build_fov_mask(img, right_strip_px=110, bottom_strip_px=15)

    lumen_mask, vesselness = segment_lumen_isolated_fixed(
        img, fov_mask, min_obj=200, close_r=3, open_r=1, ridge_dilate=3
    )

    skel, dist, coords, diam = skeleton_and_distance(lumen_mask)
    
    cv2.imwrite(f"{out_prefix}_vesselness.png", (vesselness*255).astype(np.uint8))
    cv2.imwrite(f"{out_prefix}_mask.png", (lumen_mask.astype(np.uint8))*255)

    # skeleton overlay for sanity-check
    cv2.imwrite(f"{out_prefix}_vesselness.png", (vesselness*255).astype(np.uint8))
    cv2.imwrite(f"{out_prefix}_mask.png", (lumen_mask.astype(np.uint8))*255)
    base8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR)
    ys, xs = np.nonzero(skel); rgb[ys,xs] = (255,255,255)
    cv2.imwrite(f"{out_prefix}_skeleton.png", rgb)

    G = build_graph_from_skeleton(skel)
    path, H = longest_path_on_lcc(G)
    if not path:
        raise RuntimeError("Empty skeleton graph.")

    yx_to_idx = { (r,c): k for k,(r,c) in enumerate(map(tuple, coords)) }
    path_idx = np.array([yx_to_idx[(H.nodes[n]['y'], H.nodes[n]['x'])] for n in path])

    yx_ord, d_ord, ref_ord, pds, min_idx = measure_stenosis_along_path(coords, diam, path_idx)

    if debug_plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,3))
            plt.plot(d_ord, label="Diameter (px)")
            plt.plot(ref_ord, linestyle="--", label="Ref (P90)")
            plt.title(f"Diameter profile: {os.path.basename(input_path)}")
            plt.xlabel("Centerline index")
            plt.ylabel("Diameter (px)")
            plt.legend()
            plt.tight_layout()
            plot_path = f"{out_prefix}_diam_profile.png"
            plt.savefig(plot_path, dpi=160)
            plt.close()
            print(f"[plot] saved {plot_path}")
        except Exception as e:
            print(f"[WARN] plot failed for {input_path}: {e}")


    # ---- NEW: choose which indices to mark on the overlay ----
    if mark_index is not None:     # force a specific plot index
        idxs = [int(np.clip(mark_index, 0, len(d_ord)-1))]
    elif mark_deepest:             # force the global minimum diameter
        idxs = [int(np.argmin(d_ord))]
    else:
        # default automatic detection (V-shape / sudden drop logic you added)
        idxs = pick_stenoses(d_ord, pds, rel_drop=0.30, abs_drop=1.5, window=12)
        if len(idxs) == 0 and len(d_ord) > 0:
            idxs = [int(np.argmin(d_ord))]
        center_idx = int(idxs[0])

    # expand to lesion edges using %DS threshold
    i0, i1 = lesion_region(d_ord, ref_ord, pds, center_idx, pds_thresh=50.0, max_expand=40)
    # ----------------------------------------------------------

    # draw targeted overlay
    if pds[center_idx] < 20:
        print(f"[skip] {os.path.basename(input_path)}: %DS={pds[center_idx]:.1f} < 20; not drawing.")
    else:
        overlay_path = f"{out_prefix}_overlay.png"
        draw_overlay_targeted(img, yx_ord, d_ord, ref_ord, pds, center_idx, i0, i1, overlay_path)
    # ---- NEW: add curve index column so plot ↔ pixels is explicit ----
    curve_idx = np.arange(len(d_ord))
    # ----------------------------------------------------------

    # Build detections list for overlay/CSV (keep as-is, just include idx if you like)
    detections = [{
        "idx": int(i),                      # curve index on the plot (optional but handy)
        "r": int(yx_ord[i,0]),
        "c": int(yx_ord[i,1]),
        "diam_px": float(d_ord[i]),
        "ref_px": float(ref_ord[i]),
        "pds": float(pds[i])
    } for i in idxs]

    # Per-image CSV (add curve_idx column)
    df = pd.DataFrame({
        "curve_idx": curve_idx,           # <— NEW
        "y": yx_ord[:, 0],
        "x": yx_ord[:, 1],
        "diameter_px": d_ord,
        "ref_diameter_px": ref_ord,
        "percent_diameter_stenosis": pds
    })

    if px is not None:
        df["diameter_mm"] = df["diameter_px"] * px
        df["ref_diameter_mm"] = df["ref_diameter_px"] * px

    csv_path = f"{out_prefix}.csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)

    # detections CSV + overlay
    det_csv = f"{out_prefix}_detections.csv"
    pd.DataFrame(detections).to_csv(det_csv, index=False)

    overlay_path = f"{out_prefix}_overlay.png"
    draw_overlay_targeted(img, yx_ord, d_ord, ref_ord, pds, center_idx, i0, i1, overlay_path)


    first = detections[0] if detections else {}
    summary = {
        "image": input_path,
        "overlay_path": overlay_path,
        "per_image_csv": csv_path,
        "detections_csv": det_csv if detections else "",
        "num_detections": len(detections),
        "first_x": int(first.get("c", -1)),
        "first_y": int(first.get("r", -1)),
        "first_diam_px": float(first.get("diam_px", np.nan)),
        "first_ref_px": float(first.get("ref_px", np.nan)),
        "first_%DS": float(first.get("pds", np.nan))
    }
    if px is not None and detections:
        summary.update({
            "first_diam_mm": float(first["diam_px"]) * px,
            "first_ref_mm": float(first["ref_px"]) * px
        })
    return summary

def pick_stenoses(diam, pds, rel_drop=0.4, abs_drop=2.0, window=15):
    """
    Detect sudden focal diameter drops (stenoses).
    rel_drop: fractional diameter drop (e.g. 0.4 = 40%)
    abs_drop: absolute drop in pixels required
    window: neighborhood size (samples)
    """
    d = np.asarray(diam)
    stenoses = []

    # compute derivative (slope)
    grad = np.gradient(d)

    # look for local minima (potential tight points)
    mins, _ = find_peaks(-d)

    for i in mins:
        left = max(0, i - window)
        right = min(len(d), i + window)
        d_left = np.max(d[left:i]) if i > left else d[i]
        d_right = np.max(d[i:right]) if i < right else d[i]
        drop = max(d_left - d[i], d_right - d[i])
        rel = drop / (max(d_left, d_right) + 1e-8)
        # stenosis if big relative or absolute drop and shape looks like V
        if (drop >= abs_drop or rel >= rel_drop) and grad[i-1] < 0 and grad[i+1] > 0:
            stenoses.append(i)

    # sort by severity (narrowest first)
    stenoses = sorted(stenoses, key=lambda j: d[j])
    return stenoses


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
    # also include common extensions if user left default pattern
    if pattern == "*.png":
        extra = []
        for ext in VALID_EXT:
            extra += glob.glob(os.path.join(root, "**" if recursive else "", f"*{ext}"),
                               recursive=recursive)
        paths = sorted(set(paths + extra))
    return [p for p in paths if os.path.splitext(p)[1].lower() in VALID_EXT]

def safe_stem(path):
    s = os.path.splitext(os.path.basename(path))[0]
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Centerline + diameter + stenosis on images or a folder")
    ap.add_argument("--in", dest="inp", required=True, help="input image or directory")
    ap.add_argument("--out", dest="out", required=True, help="output dir OR file prefix for single image")
    ap.add_argument("--px", dest="px", type=float, default=None, help="mm per pixel (optional)")
    ap.add_argument("--pattern", default="*.png", help="glob for folder mode (default: *.png)")
    ap.add_argument("--recursive", action="store_true", help="recurse into subfolders")
    ap.add_argument("--workers", type=int, default=1, help="parallel workers for folder mode")
    ap.add_argument("--pds_thresh", type=float, default=50.0,
               help="Flag as stenosis if %%DS >= this (default 50)")
    ap.add_argument("--topk", type=int, default=3,
               help="Max number of stenoses to report per image (default 3)")
    ap.add_argument("--suppression", type=int, default=20,
               help="Non-maximum suppression window on the profile (samples)")
    ap.add_argument("--mark_deepest", action="store_true",
                help="Force marking the global minimum diameter along the path")
    ap.add_argument("--mark_index", type=int, default=None,
                help="Force marking a specific centerline index from the diameter plot")
    ap.add_argument("--debug_plot", action="store_true",
                help="Save a diameter profile plot per image (requires matplotlib)")

    return ap.parse_args()

def main():
    args = parse_args()
    inp = args.inp
    out = args.out
    px  = args.px

    if os.path.isfile(inp):
        # single image mode
        if os.path.isdir(out):
            out_prefix = os.path.join(out, safe_stem(inp))
        else:
            out_prefix = out
        res = run_one_image(inp, out_prefix, px=px,
        pds_thresh=args.pds_thresh, topk=args.topk, suppression=args.suppression,
        mark_deepest=args.mark_deepest, mark_index=args.mark_index, debug_plot=args.debug_plot)

        print(pd.Series(res).to_string())
        return

    # folder mode
    images = list_images(inp, pattern=args.pattern, recursive=args.recursive)
    if not images:
        print("No images found. Check --pattern/--recursive.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out, exist_ok=True)

    summaries = []
    errors = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut2img = {}
            for im in images:
                pref = os.path.join(out, safe_stem(im))
                fut = ex.submit(run_one_image, im, pref, px)
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
                summaries.append(run_one_image(im, pref, px,
                args.pds_thresh, args.topk, args.suppression,
                args.mark_deepest, args.mark_index, args.debug_plot))
                print(f"[OK] {im}")
            except Exception as e:
                errors.append((im, str(e)))
                print(f"[ERR] {im}: {e}", file=sys.stderr)

    # write master summary
    if summaries:
        df = pd.DataFrame(summaries)
        df.to_csv(os.path.join(out, "summary.csv"), index=False)
        print(f"\nSaved master summary: {os.path.join(out, 'summary.csv')}")
        if px is not None:
            print("Units: diameters also reported in mm using --px.")
    if errors:
        print("\nSome images failed:", file=sys.stderr)
        for im, msg in errors:
            print(f"- {im}: {msg}", file=sys.stderr)

if __name__ == "__main__":
    main()
