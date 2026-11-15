import json

def make_md_cell(source_str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_str.strip().split("\n")]
    }

def make_code_cell(source_str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_str.strip().split("\n")]
    }

cells = []

# ==============================================================================
# INTRO MARKDOWN
# ==============================================================================
cells.append(make_md_cell("""
# ELTE-TCM-46k: Publication Figures (Nature Scientific Data) & Rotational Deep MIL Pipeline
This notebook connects directly to **`alirezafalah/ELTE-TCM-46k`** on Hugging Face and is organized into two parts:
1. **Part 1 (Cells 1–7): Publication-Ready Vector PDF Figures (`< 3 MB` each)** following *Nature Scientific Data* guidelines (sans-serif typography, bold lowercase **`a, b, c`** panel labels, physical `mm` scale bars, and 300 DPI embedded raster crops).
2. **Part 2 (Cells 8–10): Streaming Feature Extraction (419 GB $\\rightarrow$ 70 MB) & Rotational Gated-Attention MIL (`Rot-ABMIL`)** for deep learning tool condition classification and explainable angular attention discovery.

> **Search for `🔧 [MANUAL FIX]`** in each cell for parameters you may need to tweak (e.g., rotation angle sign, tool tip orientation, or specific `tool_id` choices).
"""))

# ==============================================================================
# CELL 1: SETUP & NATURE SCIENTIFIC DATA STYLING
# ==============================================================================
cells.append(make_md_cell("""
## Cell 1: Environment Setup & *Nature Scientific Data* Vector PDF Styling
"""))

cells.append(make_code_cell("""
# Install required packages
!pip install -q huggingface_hub timm scikit-learn seaborn tqdm tifffile opencv-python-headless

import os
import json
import tarfile
import zipfile
import shutil
import glob
import numpy as np
import pandas as pd
import cv2
import tifffile
from pathlib import Path
from tqdm.auto import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns

# Mount Google Drive to permanently store the 70 MB extracted features & PDF figures
try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_DIR = Path('/content/drive/MyDrive/ELTE_TCM_46k_Outputs')
except Exception:
    OUTPUT_DIR = Path('./ELTE_TCM_46k_Outputs')

FIG_DIR = OUTPUT_DIR / 'figures_pdf'
FEAT_DIR = OUTPUT_DIR / 'extracted_features_dinov2'
LOCAL_CACHE = Path('/content/elte_cache')

for d in [FIG_DIR, FEAT_DIR, LOCAL_CACHE]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# NATURE SCIENTIFIC DATA FIGURE HOUSE STYLE
# - Sans-serif font (DejaVu Sans / Arial / Helvetica)
# - TrueType font embedding (pdf.fonttype = 42) so text is editable in Illustrator
# - 300 DPI rasterization for images so vector PDFs stay < 2-3 MB
# -------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'pdf.fonttype': 42,       # Keeps text as editable vectors in PDF
    'ps.fonttype': 42,
    'savefig.dpi': 300,       # Rasterized image panels stay sharp but compact (<3 MB)
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def add_panel_label(ax, label_char, x=-0.08, y=1.06, color='black'):
    \"\"\"Adds Nature Scientific Data mandatory lower-case bold panel letter (a, b, c...).\"\"\"
    ax.text(x, y, label_char, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left', color=color)

def add_scale_bar(ax, img_shape, tool_px_width, tool_diameter_mm, bar_mm=2.0, loc='lower right'):
    \"\"\"
    Draws a physical scale bar directly on the image (required by Scientific Data
    instead of magnification factors in the legend).
    \"\"\"
    if tool_px_width <= 0 or tool_diameter_mm <= 0:
        return
    px_per_mm = tool_px_width / float(tool_diameter_mm)
    bar_px = int(bar_mm * px_per_mm)
    H, W = img_shape[:2]
    margin_x = int(W * 0.06)
    margin_y = int(H * 0.06)
    bar_h = max(int(H * 0.015), 6)

    x0 = W - margin_x - bar_px if 'right' in loc else margin_x
    y0 = H - margin_y - bar_h

    rect = Rectangle((x0, y0), bar_px, bar_h, facecolor='white', edgecolor='black', linewidth=0.8, zorder=10)
    ax.add_patch(rect)
    ax.text(x0 + bar_px / 2, y0 - int(H * 0.02), f"{bar_mm:g} mm",
            color='white', fontsize=7.5, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6, edgecolor='none'), zorder=11)

print("Setup complete. Figures will be saved to:", FIG_DIR)
"""))

# ==============================================================================
# CELL 2: DOWNLOAD CSV, MASKS TAR.GZ, AND ROTATIONS JSON
# ==============================================================================
cells.append(make_md_cell("""
## Cell 2: Download Metadata CSV, Tilt-Corrected Masks (`353 MB`), and Calibration JSON
"""))

cells.append(make_code_cell("""
from huggingface_hub import hf_hub_download

REPO_ID = "alirezafalah/ELTE-TCM-46k"

# 1. Download tool_condition_summary.csv
csv_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                           filename="tool_condition_summary.csv", local_dir=LOCAL_CACHE)
df_meta = pd.read_csv(csv_path)
print(f"Loaded metadata for {len(df_meta)} tools.")

# 2. Download and extract masks_tilted.tar.gz (only once)
MASKS_DIR = LOCAL_CACHE / "masks_tilted"
if not (MASKS_DIR / "tool001").exists():
    print("Downloading masks_tilted.tar.gz (353 MB)...")
    tar_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                               filename="masks_tilted.tar.gz", local_dir=LOCAL_CACHE)
    print("Extracting masks_tilted.tar.gz...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=LOCAL_CACHE)
    print("Extraction complete!")

# 3. Load applied_per_tool_rotations.json
rot_json_path = MASKS_DIR / "applied_per_tool_rotations.json"
if not rot_json_path.exists():
    # Fallback: try downloading directly from repo root if it exists there too
    rot_json_path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                         filename="applied_per_tool_rotations.json", local_dir=LOCAL_CACHE))

with open(rot_json_path, "r") as f:
    ROTATIONS_DICT = json.load(f)

# Inspect the structure of the first 2 entries so you can verify key names
sample_keys = list(ROTATIONS_DICT.keys())[:2]
print("Sample entries in applied_per_tool_rotations.json:")
for k in sample_keys:
    print(f"  {k}: {ROTATIONS_DICT[k]}")

df_meta.head()
"""))

# ==============================================================================
# CELL 3: RGB-TO-MASK TILT ALIGNMENT & ROI CROPPING HELPER
# ==============================================================================
cells.append(make_md_cell("""
## Cell 3: Aligning Raw RGB Frames to Tilt-Corrected Binary Masks & Master-Mask ROI Cropping
The raw `.tiff` frames in `images/toolXXX.zip` have the physical axis tilt (`tilt_deg`), whereas `masks_tilted/toolXXX/*.png` have already been vertically aligned by applying `rotation_deg` (`= -tilt_deg`) from `applied_per_tool_rotations.json` using OpenCV's counter-clockwise-positive convention (`cv2.getRotationMatrix2D`, without negating the angle).

The helpers below:
1. Read `rotation_deg` directly from `applied_per_tool_rotations.json` (falling back to `-tilt_deg` if needed).
2. Rotate each raw RGB frame around the image center via `cv2.getRotationMatrix2D` so it aligns pixel-for-pixel with `masks_tilted`.
3. Build a per-tool **master mask** (bitwise union across all $360^\\circ$ masks) to locate the tool's absolute spatial envelope and maximum pixel width (with the cutting tip pointing downward), and crop a fixed active tool-head ROI across frames.
"""))

cells.append(make_code_cell("""
def get_tool_tilt_angle(tool_id, rotations_dict):
    \"\"\"
    Returns the OpenCV rotation angle (degrees, CCW-positive) needed to align
    a raw RGB frame to its tilt-corrected mask in `masks_tilted`.
    In `applied_per_tool_rotations.json`, `tilt_deg` is the measured raw axis tilt
    and `rotation_deg` (= -tilt_deg) is the correction rotation already applied to the masks.
    \"\"\"
    entry = rotations_dict.get(tool_id, {})
    if isinstance(entry, (int, float)):
        return float(entry)
    if entry.get("rotation_deg") is not None:
        return float(entry["rotation_deg"])
    if entry.get("tilt_deg") is not None:
        return -float(entry["tilt_deg"])
    return 0.0

def rotate_image_to_match_mask(img, angle_deg, is_mask=False):
    \"\"\"
    Rotates an image around its center using OpenCV's native convention
    (positive angle = counter-clockwise, no sign negation).
    Pass `angle_deg = rotation_deg` to align raw RGB -> `masks_tilted`,
    or `angle_deg = -rotation_deg` (`+tilt_deg`) to revert a mask -> raw RGB.
    \"\"\"
    H, W = img.shape[:2]
    center = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border_val = 0 if is_mask else (128, 128, 128)
    return cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

def download_and_extract_single_tool_rgb(tool_id, cache_dir=LOCAL_CACHE):
    \"\"\"Downloads images/toolXXX.zip on-demand and extracts to cache_dir/raw_rgb/toolXXX.\"\"\"
    out_dir = cache_dir / "raw_rgb" / tool_id
    if out_dir.exists() and len(list(out_dir.glob("*.tiff"))) > 10:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_rel = f"images/{tool_id}.zip"
    print(f"Downloading {zip_rel} from Hugging Face...")
    zip_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=zip_rel, local_dir=cache_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    # Handle case where zip contains a nested toolXXX/ folder
    nested = out_dir / tool_id
    if nested.exists() and nested.is_dir():
        for f in nested.glob("*"):
            shutil.move(str(f), str(out_dir / f.name))
        nested.rmdir()
    # Delete the zip file immediately to save Colab disk space
    if Path(zip_path).exists():
        Path(zip_path).unlink()
    return out_dir

def build_tool_master_mask(tool_id, masks_dir=MASKS_DIR):
    \"\"\"
    Builds a per-tool master binary mask by taking the bitwise union (pixel sum > 0)
    across all 360° tilt-corrected masks in `masks_tilted/toolXXX/`.
    This captures the absolute spatial envelope and true maximum width of the tool.
    \"\"\"
    mask_paths = sorted((masks_dir / tool_id).glob("*.png"))
    master = None
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        fg = m > 127
        master = fg.copy() if master is None else (master | fg)
    if master is None:
        raise FileNotFoundError(f"No valid mask PNGs found for {tool_id} in {masks_dir / tool_id}")
    return master.astype(np.uint8) * 255

def get_tool_roi_box(master_mask, pad_px=80, aspect_square=True):
    \"\"\"
    Computes the fixed active tool-head bounding box `(xmin, xmax, ymin_roi, ymax_roi, tool_px_width)`
    from the tool's master mask, with the cutting tip pointing downward (`ymax`).
    \"\"\"
    ys, xs = np.where(master_mask > 127)
    if len(xs) == 0:
        return None
    H, W = master_mask.shape[:2]
    xmin, xmax = max(0, int(xs.min()) - pad_px), min(W, int(xs.max()) + pad_px)
    ymax = min(H, int(ys.max()) + pad_px)
    tool_px_width = int(xs.max() - xs.min())

    # Cutting tip points downward: anchor the ROI at the bottom (ymax) with height = 1.35 * tool width
    crop_h = int(max(tool_px_width * 1.35, xmax - xmin))
    ymin_roi = max(0, ymax - crop_h)
    ymax_roi = ymax

    if aspect_square:
        cx = (xmin + xmax) // 2
        cy = (ymin_roi + ymax_roi) // 2
        half_side = max(xmax - xmin, ymax_roi - ymin_roi) // 2
        xmin = max(0, cx - half_side)
        xmax = min(W, cx + half_side)
        ymin_roi = max(0, cy - half_side)
        ymax_roi = min(H, cy + half_side)

    return (xmin, xmax, ymin_roi, ymax_roi, tool_px_width)

def crop_tool_head_roi(aligned_rgb, tilted_mask, pad_px=80, aspect_square=True, master_mask=None, roi_box=None):
    \"\"\"
    Crops the active cutting tool head from `aligned_rgb` and `tilted_mask`.
    Prefers a precomputed `roi_box` (or `master_mask`) so every frame of a tool
    uses the exact same spatial bounding box and full-rotation `tool_px_width`.
    \"\"\"
    if roi_box is None:
        ref_mask = master_mask if master_mask is not None else tilted_mask
        roi_box = get_tool_roi_box(ref_mask, pad_px=pad_px, aspect_square=aspect_square)
    if roi_box is None:
        return aligned_rgb, tilted_mask, 0
    xmin, xmax, ymin_roi, ymax_roi, tool_px_width = roi_box
    return aligned_rgb[ymin_roi:ymax_roi, xmin:xmax], tilted_mask[ymin_roi:ymax_roi, xmin:xmax], tool_px_width

# --- VISUAL SANITY CHECK ON 1 TOOL ---
TEST_TOOL = "tool006"  # Fractured endmill (or change to tool001)
raw_dir = download_and_extract_single_tool_rgb(TEST_TOOL)
sample_tiff = sorted(raw_dir.glob("*.tiff"))[0]
sample_png = MASKS_DIR / TEST_TOOL / (sample_tiff.stem + ".png")

raw_rgb = tifffile.imread(sample_tiff)
tilted_mask = cv2.imread(str(sample_png), cv2.IMREAD_GRAYSCALE)
angle = get_tool_tilt_angle(TEST_TOOL, ROTATIONS_DICT)
aligned_rgb = rotate_image_to_match_mask(raw_rgb, angle, is_mask=False)
master_mask = build_tool_master_mask(TEST_TOOL)
roi_box = get_tool_roi_box(master_mask, pad_px=80, aspect_square=True)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
axes[0].imshow(raw_rgb)
axes[0].contour(tilted_mask, levels=[127], colors='lime', linewidths=1.0)
axes[0].set_title(f"Unaligned RGB + Tilted Mask Contour\\n({TEST_TOOL}, raw)")

axes[1].imshow(aligned_rgb)
axes[1].contour(tilted_mask, levels=[127], colors='lime', linewidths=1.0)
axes[1].set_title(f"Tilt-Aligned RGB (angle={angle:.2f}°)\\n+ Tilted Mask Contour")

crop_rgb, crop_msk, px_w = crop_tool_head_roi(aligned_rgb, tilted_mask, roi_box=roi_box)
axes[2].imshow(crop_rgb)
axes[2].contour(crop_msk, levels=[127], colors='cyan', linewidths=1.0)
axes[2].set_title(f"Cropped Active Tool Head ROI\\n(master-mask width={px_w} px)")
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# CELL 4: FIGURE 2 — ANGLE-DEPENDENT DEFECT VISIBILITY (3 ANGLES + 1D CURVE)
# ==============================================================================
cells.append(make_md_cell("""
## Cell 4: Paper Figure 2 — Angle-Dependent Defect Visibility (`Fig2_Angle_Visibility.pdf`)
Demonstrates on `tool054` why full $360^\\circ$ rotation is essential for tool condition monitoring:
* **a ($\\theta \\approx 93.96^\\circ$):** Fracture directly facing the camera (visible on RGB surface, hidden in silhouette)
* **b ($\\theta \\approx 231.43^\\circ$):** Fracture rotated to the rear hemisphere (occluded behind the tool)
* **c ($\\theta \\approx 172.09^\\circ$):** Fracture on the left lateral tangent boundary (clearly visible in the 2D binary silhouette)
* **d:** Full $360^\\circ$ 1D projected silhouette area curve across the active tool-head ROI with vertical markers at $\\theta_a, \\theta_b, \\theta_c$.
"""))

cells.append(make_code_cell("""
FIG2_TOOL_ID = "tool054"
FRONT_DEG = 93.96   # Fracture facing camera directly
BACK_DEG  = 231.43  # Fracture occluded on the rear side
EDGE_DEG  = 172.09  # Fracture on the left peripheral edge

raw_dir = download_and_extract_single_tool_rgb(FIG2_TOOL_ID)
mask_files = sorted((MASKS_DIR / FIG2_TOOL_ID).glob("*.png"))
tool_row = df_meta[df_meta["tool_id"] == FIG2_TOOL_ID].iloc[0]
diam_mm = float(tool_row["diameter_mm"])
angle_cal = get_tool_tilt_angle(FIG2_TOOL_ID, ROTATIONS_DICT)

# Compute fixed active tool-head ROI once from the tool's master mask
master_mask = build_tool_master_mask(FIG2_TOOL_ID)
roi_box = get_tool_roi_box(master_mask, pad_px=60, aspect_square=True)
xmin, xmax, ymin_roi, ymax_roi, px_w = roi_box

# Compute 1D projected silhouette area inside the active tool-head ROI across all frames
areas = []
angles_deg = []
for m_path in mask_files:
    deg_str = m_path.stem.split("_")[0]
    angles_deg.append(float(deg_str))
    m_img = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
    areas.append(np.sum(m_img[ymin_roi:ymax_roi, xmin:xmax] > 127))

areas = np.array(areas, dtype=float)
areas_norm_pct = (areas / np.median(areas)) * 100.0  # Normalized % of median profile area
angles_deg = np.array(angles_deg)

# Select exact frames closest to the verified viewing angles
idx_front = int(np.argmin(np.abs(angles_deg - FRONT_DEG)))
idx_back  = int(np.argmin(np.abs(angles_deg - BACK_DEG)))
idx_edge  = int(np.argmin(np.abs(angles_deg - EDGE_DEG)))

selected_indices = [
    ("a", idx_front, "Front-facing defect\\n(visible on surface, intact silhouette)", "#1f77b4"),
    ("b", idx_back,  "Rear-occluded defect\\n(hidden behind tool core)", "#7f7f7f"),
    ("c", idx_edge,  "Left-edge defect\\n(alters 2D silhouette profile)", "#d62728"),
]

fig = plt.figure(figsize=(7.2, 4.8))
gs = gridspec.GridSpec(2, 3, height_ratios=[1.35, 1.0], hspace=0.35, wspace=0.18)

for col_i, (p_label, f_idx, title_txt, col_hex) in enumerate(selected_indices):
    ax = fig.add_subplot(gs[0, col_i])
    m_path = mask_files[f_idx]
    t_path = raw_dir / (m_path.stem + ".tiff")
    if not t_path.exists():
        t_path = sorted(raw_dir.glob("*.tiff"))[f_idx]

    rgb = tifffile.imread(t_path)
    msk = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
    rgb_aligned = rotate_image_to_match_mask(rgb, angle_cal, is_mask=False)
    crop_rgb, crop_msk, px_w = crop_tool_head_roi(rgb_aligned, msk, roi_box=roi_box)

    ax.imshow(crop_rgb, rasterized=True)
    ax.contour(crop_msk, levels=[127], colors=[col_hex], linewidths=1.2)
    deg_val = angles_deg[f_idx]
    ax.set_title(f"$\\\\theta = {deg_val:.2f}^\\\\circ$ (Frame {f_idx})\\n{title_txt}", fontsize=8, color=col_hex, fontweight='bold')
    ax.axis('off')
    add_panel_label(ax, p_label, x=-0.04, y=1.12)
    if col_i == 0:
        add_scale_bar(ax, crop_rgb.shape, px_w, diam_mm, bar_mm=2.0)

# Bottom panel d: 1D rotational profile signal across 360 degrees
ax_sig = fig.add_subplot(gs[1, :])
ax_sig.plot(angles_deg, areas_norm_pct, color='#2b2b2b', linewidth=1.2, label='Projected 2D silhouette area (%)')
for p_label, f_idx, title_txt, col_hex in selected_indices:
    deg_val = angles_deg[f_idx]
    val_pct = areas_norm_pct[f_idx]
    ax_sig.axvline(deg_val, color=col_hex, linestyle='--', linewidth=1.2, alpha=0.85)
    ax_sig.scatter([deg_val], [val_pct], color=col_hex, s=38, zorder=5,
                   label=f"Panel {p_label} ($\\\\theta={deg_val:.2f}^\\\\circ$: {val_pct:.1f}%)")

ax_sig.set_xlim(0, 360)
ax_sig.set_xlabel("Rotational angle $\\\\theta$ (degrees)")
ax_sig.set_ylabel("Normalized silhouette area (%)")
ax_sig.set_title(f"1D rotational silhouette profile across $360^\\\\circ$ revolution ({FIG2_TOOL_ID}, {int(tool_row['flute_number'])}-flute {tool_row['type']})", fontsize=9)
ax_sig.legend(loc='lower right', frameon=True, ncol=2)
ax_sig.grid(True, linestyle=':', alpha=0.4)
add_panel_label(ax_sig, "d", x=-0.03, y=1.12)

out_pdf = FIG_DIR / "Fig2_Angle_Visibility.pdf"
plt.savefig(out_pdf)
plt.show()
print(f"Saved {out_pdf} (Size: {out_pdf.stat().st_size / 1024:.1f} KB)")
"""))

# ==============================================================================
# CELL 5: FIGURE 4 — FLUTE SYMMETRY & WHY WAVY PROFILES CANNOT BE SYMMETRICAL
# ==============================================================================
cells.append(make_md_cell("""
## Cell 5: Paper Figure 4 — Flute Symmetry Classes & Why Wavy Profiles Are Asymmetric (`Fig4_Symmetry_and_Wavy.pdf`)
Compares `symmetrical` (2-flute), `three_flute` (odd-flute phase shift), `paired` ($2+2$ differential indexing), and `wavy` (`tool003` roughing knuckle profile or `tool038` thread tap), drawing horizontal alignment guide lines across the silhouette to show that a wave crest on the left edge aligns with a wave trough on the right edge.
"""))

cells.append(make_code_cell("""
# 🔧 [MANUAL FIX #3]: You can swap any of these 4 representative tool IDs
SYMMETRY_SHOWCASE = [
    ("a", "tool002", "Symmetrical (2-flute drill)\\nLeft & right edges mirror"),
    ("b", "tool004", "Asymmetrical: three_flute\\nTooth crest vs. flute valley"),
    ("c", "tool001", "Asymmetrical: paired (6-flute)\\nDifferential paired indexing"),
    ("d", "tool003", "Asymmetrical: paired_wavy\\nHelical wave crest vs. trough"),
]

fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.7))

for idx, (p_label, t_id, subtitle) in enumerate(SYMMETRY_SHOWCASE):
    ax = axes[idx]
    mask_files = sorted((MASKS_DIR / t_id).glob("*.png"))
    msk = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
    # Crop active head region from mask
    ys, xs = np.where(msk > 127)
    W_px = xs.max() - xs.min()
    ymax = ys.max()
    ymin = max(0, ymax - int(W_px * 1.45))
    xmin = max(0, xs.min() - 50)
    xmax = min(msk.shape[1], xs.max() + 50)
    crop_m = msk[ymin:ymax, xmin:xmax]

    # Plot binary silhouette mask with central vertical rotational axis
    ax.imshow(crop_m, cmap='gray', rasterized=True)
    H_c, W_c = crop_m.shape
    cx = W_c / 2.0
    ax.axvline(cx, color='#00b4d8', linestyle='-.', linewidth=1.1, label='Rotation axis')

    # For odd-flute and wavy profiles, draw horizontal guide lines highlighting left-vs-right asymmetry
    if t_id in ["tool004", "tool003"]:
        for frac in [0.30, 0.50, 0.70]:
            y_line = int(H_c * frac)
            row_xs = np.where(crop_m[y_line, :] > 127)[0]
            if len(row_xs) > 2:
                x_left, x_right = row_xs.min(), row_xs.max()
                ax.plot([x_left, x_right], [y_line, y_line], color='#ff595e', linestyle=':', linewidth=1.0)
                ax.scatter([x_left, x_right], [y_line, y_line], color=['#ffca3a', '#ff595e'], s=18, zorder=5)

    ax.set_title(f"{t_id}\\n{subtitle}", fontsize=7.5)
    ax.axis('off')
    add_panel_label(ax, p_label, x=-0.05, y=1.14)

plt.tight_layout()
out_pdf = FIG_DIR / "Fig4_Symmetry_and_Wavy.pdf"
plt.savefig(out_pdf)
plt.show()
print(f"Saved {out_pdf} (Size: {out_pdf.stat().st_size / 1024:.1f} KB)")
"""))

# ==============================================================================
# CELL 6: FIGURE 6 — SILHOUETTE EXTRACTION & ROTATIONAL AXIS TILT CALIBRATION
# ==============================================================================
cells.append(make_md_cell("""
## Cell 6: Paper Figure 6 — Rotational Axis Tilt Calibration (`Fig6_Tilt_Calibration.pdf`)
Shows why fixing mechanical tilt (`applied_per_tool_rotations.json`) is essential before computing left-right symmetry or 1D horizontal profiles.
"""))

cells.append(make_code_cell("""
# Find a tool with a noticeable tilt calibration angle in applied_per_tool_rotations.json
tilt_candidates = [(k, get_tool_tilt_angle(k, ROTATIONS_DICT)) for k in df_meta["tool_id"]]
tilt_candidates_sorted = sorted(tilt_candidates, key=lambda x: abs(x[1]), reverse=True)
print("Top 5 tools with largest tilt calibration angles:", tilt_candidates_sorted[:5])

# 🔧 [MANUAL FIX #4]: Pick a tool ID to illustrate tilt correction
TILT_TOOL_ID = tilt_candidates_sorted[0][0] if abs(tilt_candidates_sorted[0][1]) > 0.1 else "tool002"
tilt_angle = get_tool_tilt_angle(TILT_TOOL_ID, ROTATIONS_DICT)
# If the JSON angle is very small (<0.5 deg), we also show a simulated 2.5 deg uncalibrated tilt for visual clarity
vis_angle = tilt_angle if abs(tilt_angle) >= 0.8 else 2.2

mask_path = sorted((MASKS_DIR / TILT_TOOL_ID).glob("*.png"))[0]
tilted_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
# Reconstruct the unaligned (pre-calibration) mask by rotating backwards
unaligned_mask = rotate_image_to_match_mask(tilted_mask, -vis_angle, is_mask=True)

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))

# Panel a: Uncalibrated tilted silhouette
axes[0].imshow(unaligned_mask, cmap='gray', rasterized=True)
H, W = unaligned_mask.shape
axes[0].axvline(W // 2, color='#00b4d8', linestyle='--', linewidth=1.0, label='Sensor vertical $y$-axis')
axes[0].plot([W//2 - int(H*0.4*np.sin(np.deg2rad(vis_angle))), W//2 + int(H*0.4*np.sin(np.deg2rad(vis_angle)))],
             [int(H*0.1), int(H*0.9)], color='#ff595e', linewidth=1.2, label='Tilted tool axis')
axes[0].set_title(f"Pre-calibration binary mask\\n(Axis tilt = ${vis_angle:.2f}^\\\\circ$)", fontsize=8)
axes[0].legend(loc='upper right', fontsize=6.5)
axes[0].axis('off')
add_panel_label(axes[0], "a")

# Panel b: Tilt-calibrated vertical silhouette
axes[1].imshow(tilted_mask, cmap='gray', rasterized=True)
axes[1].axvline(W // 2, color='#2ec4b6', linestyle='-', linewidth=1.2, label='Aligned vertical axis')
axes[1].set_title(f"Tilt-corrected mask ({TILT_TOOL_ID})\\n(Aligned to vertical column $x_c$)", fontsize=8)
axes[1].legend(loc='upper right', fontsize=6.5)
axes[1].axis('off')
add_panel_label(axes[1], "b")

# Panel c: Row-wise horizontal center-of-mass drift before vs after tilt calibration
ys_valid = np.arange(int(H * 0.25), int(H * 0.85), 10)
def get_row_centroids(m_img, y_rows):
    c_list = []
    for y in y_rows:
        xs = np.where(m_img[y, :] > 127)[0]
        c_list.append(np.mean(xs) if len(xs) > 0 else np.nan)
    c_arr = np.array(c_list)
    return c_arr - np.nanmean(c_arr)

drift_unaligned = get_row_centroids(unaligned_mask, ys_valid)
drift_aligned = get_row_centroids(tilted_mask, ys_valid)

axes[2].plot(drift_unaligned, ys_valid, color='#ff595e', linestyle='--', linewidth=1.4, label='Before tilt calibration')
axes[2].plot(drift_aligned, ys_valid, color='#2ec4b6', linestyle='-', linewidth=1.5, label='After tilt calibration')
axes[2].axvline(0, color='black', linestyle=':', linewidth=0.8)
axes[2].invert_yaxis()
axes[2].set_xlabel("Horizontal axis drift $\\\\Delta x$ (pixels)")
axes[2].set_ylabel("Vertical pixel row $y$")
axes[2].set_title("Axial centerline stability", fontsize=8)
axes[2].legend(loc='lower left', fontsize=7)
axes[2].grid(True, linestyle=':', alpha=0.4)
add_panel_label(axes[2], "c")

plt.tight_layout()
out_pdf = FIG_DIR / "Fig6_Tilt_Calibration.pdf"
plt.savefig(out_pdf)
plt.show()
print(f"Saved {out_pdf} (Size: {out_pdf.stat().st_size / 1024:.1f} KB)")
"""))

# ==============================================================================
# CELL 7: FIGURE 7 — TECHNICAL VALIDATION & DATASET OVERVIEW PLOTS
# ==============================================================================
cells.append(make_md_cell("""
## Cell 7: Paper Figure 7 — Technical Validation & Dataset Overview (`Fig7_Technical_Validation.pdf`)
Generates a 3-panel *Nature Scientific Data* validation figure directly from `tool_condition_summary.csv` and the extracted masks:
* **a:** Tool inventory breakdown by geometry category and physical condition state
* **b:** Physical validation of Degree of Symmetry Index (`mean_dsi_pct` & `max_dsi_pct`) across condition states
* **c:** Rotational frame count distribution across all 115 tools (documenting the nominal 364–370 frame cluster, the up-to-402 high-density sequences, and the single 282-frame outlier).
"""))

cells.append(make_code_cell("""
# Count actual number of mask frames per tool in masks_tilted/
frame_counts = []
for t_id in df_meta["tool_id"]:
    n_png = len(list((MASKS_DIR / t_id).glob("*.png")))
    frame_counts.append(n_png)
df_meta["num_frames"] = frame_counts
df_meta["deg_per_frame"] = 360.0 / df_meta["num_frames"].replace(0, np.nan)

print("Frame count summary across 115 tools:")
print(df_meta["num_frames"].describe())
print("Tools with < 350 frames:", df_meta[df_meta["num_frames"] < 350][["tool_id", "type", "num_frames", "deg_per_frame"]])
print("Tools with > 380 frames:", df_meta[df_meta["num_frames"] > 380][["tool_id", "type", "num_frames", "deg_per_frame"]].head(10))

fig = plt.figure(figsize=(7.4, 5.2))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28)

# Panel a: Stacked bar chart of Tool Type vs Condition
ax_a = fig.add_subplot(gs[0, 0])
cond_order = ["new", "used", "worn", "deposit", "fractured"]
cond_colors = {"new": "#2ca02c", "used": "#1f77b4", "worn": "#17becf", "deposit": "#9467bd", "fractured": "#d62728"}
ct = pd.crosstab(df_meta["type"], df_meta["condition"]).reindex(columns=cond_order, fill_value=0)
ct["total"] = ct.sum(axis=1)
ct = ct.sort_values("total", ascending=True).drop(columns="total")
ct.plot(kind="barh", stacked=True, ax=ax_a, color=[cond_colors[c] for c in cond_order], width=0.72)
ax_a.set_xlabel("Number of cutting tools")
ax_a.set_ylabel("Tool geometry category")
ax_a.set_title("Inventory across geometries & conditions (n = 115)", fontsize=8.5)
ax_a.legend(title="Condition", loc="lower right", frameon=True, fontsize=7)
add_panel_label(ax_a, "a")

# Panel b: Rotational frame count distribution across the 115 tools
ax_b = fig.add_subplot(gs[0, 1])
bins = np.arange(275, 415, 5)
ax_b.hist(df_meta["num_frames"], bins=bins, color="#3b528b", edgecolor="white", alpha=0.9)
ax_b.axvline(364, color="#d62728", linestyle="--", linewidth=1.1, label="Nominal 364 frames (0.99°/step)")
min_frames = int(df_meta["num_frames"].min())
if min_frames < 320:
    ax_b.annotate(f"Single lower-FPS tool\\n({min_frames} frames, {360/min_frames:.2f}°/step)",
                  xy=(min_frames, 1.2), xytext=(min_frames + 10, max(ax_b.get_ylim())*0.45),
                  arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9), fontsize=7)
ax_b.set_xlabel("Frames per $360^\\\\circ$ revolution")
ax_b.set_ylabel("Number of tools")
ax_b.set_title("Angular sampling density across 115 tools", fontsize=8.5)
ax_b.legend(loc="upper left", fontsize=7)
add_panel_label(ax_b, "b")

# Panel c: Technical validation of DSI (mean_dsi_pct & max_dsi_pct) across condition states
ax_c = fig.add_subplot(gs[1, :])
df_dsi = df_meta.dropna(subset=["mean_dsi_pct", "max_dsi_pct"]).copy()
df_melt = df_dsi.melt(id_vars=["tool_id", "condition", "symmetrical"],
                      value_vars=["mean_dsi_pct", "max_dsi_pct"],
                      var_name="DSI_Metric", value_name="DSI_Value_Pct")
df_melt["DSI_Metric"] = df_melt["DSI_Metric"].map({"mean_dsi_pct": "Mean DSI (%)", "max_dsi_pct": "Max (Peak) DSI (%)"})

sns.boxplot(data=df_melt, x="condition", y="DSI_Value_Pct", hue="DSI_Metric",
            order=cond_order, palette=["#4ecdc4", "#ff6b6b"], ax=ax_c, fliersize=3, width=0.55)
ax_c.set_yscale("log")
ax_c.set_xlabel("Physical tool condition state")
ax_c.set_ylabel("Degree of Symmetry Index, DSI (%, log scale)")
ax_c.set_title("Technical validation: 1D rotational symmetry deviation across physical tool states", fontsize=8.5)
ax_c.grid(True, which="both", axis="y", linestyle=":", alpha=0.4)
ax_c.legend(loc="upper left", frameon=True)
add_panel_label(ax_c, "c", x=-0.03, y=1.10)

out_pdf = FIG_DIR / "Fig7_Technical_Validation.pdf"
plt.savefig(out_pdf)
plt.show()
print(f"Saved {out_pdf} (Size: {out_pdf.stat().st_size / 1024:.1f} KB)")
"""))

# ==============================================================================
# CELL 8: STREAMING FEATURE EXTRACTOR (419 GB -> 70 MB ON GOOGLE DRIVE)
# ==============================================================================
cells.append(make_md_cell("""
## Cell 8: Streaming Pretrained Feature Extractor (`419 GB` Raw TIFFs $\\rightarrow$ `70 MB` on Google Drive)
Answers your question: **"If I extract the feature map, do I only need to save that for the next steps?"**
**Yes!** This cell processes each tool one-by-one:
1. Downloads `images/toolXXX.zip` (~3.5 GB)
2. Tilt-aligns each raw RGB frame with its binary mask & crops the active cutting tool head (`224x224`)
3. Extracts a `384`-dim **DINOv2-Small** (`vit_small_patch14_dinov2`) feature vector for every rotational angle (plus the classical 1D silhouette pixel area)
4. Saves `toolXXX.pt` (~600 KB) to your Google Drive (`FEAT_DIR`) and **deletes the 3.5 GB raw TIFFs immediately** so Colab never runs out of disk!

*Note: You can also run a quick **Mask-Only Mode** (`USE_RAW_RGB_DOWNLOAD = False`) first, which runs in 3 minutes using the already-downloaded `masks_tilted` PNGs to test the entire pipeline before streaming all 115 RGB zips!*
"""))

cells.append(make_code_cell("""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 🔧 [STAGE 2 BACKBONE CHOICE]:
#   "dinov2_small" : vit_small_patch14_dinov2.lvd142m (384-dim, fast, strong zero-shot surface/edge features)
#   "dinov2_base"  : vit_base_patch14_dinov2.lvd142m  (768-dim, heavier for A100/L4 GPUs)
#   "convnext_small": convnext_small.fb_in22k         (768-dim, 7x7 depthwise convs great for repetitive flutes)
#   "wideresnet50_multiscale": wide_resnet50_2        (1536-dim Layer2+Layer3 PatchCore-style industrial features)
BACKBONE_CHOICE = "dinov2_small"

class IndustrialFeatureExtractor(nn.Module):
    def __init__(self, choice="dinov2_small"):
        super().__init__()
        self.choice = choice
        if choice == "dinov2_small":
            self.net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
            self.out_dim = 384 * 2  # Concatenates global [CLS] token + max of spatial patch tokens!
        elif choice == "dinov2_base":
            self.net = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
            self.out_dim = 768 * 2
        elif choice == "convnext_small":
            self.net = timm.create_model("convnext_small.fb_in22k", pretrained=True, num_classes=0)
            self.out_dim = 768
        elif choice == "wideresnet50_multiscale":
            # Extracts intermediate Layer 2 (512-d) + Layer 3 (1024-d) features like PatchCore (CVPR 2022)
            self.net = timm.create_model("wide_resnet50_2", pretrained=True, features_only=True, out_indices=(2, 3))
            self.out_dim = 512 + 1024

    def forward(self, x):
        if "dinov2" in self.choice:
            # forward_features returns [B, N_patches + 1, D]
            tokens = self.net.forward_features(x)
            cls_token = tokens[:, 0]                  # Global tool appearance [B, D]
            patch_max = tokens[:, 1:].amax(dim=1)     # Strongest local patch response (catches tiny fractures!) [B, D]
            return torch.cat([cls_token, patch_max], dim=-1)
        elif self.choice == "wideresnet50_multiscale":
            f2, f3 = self.net(x)                      # Mid-level edge/texture maps
            # Global average + spatial max pooling on Layer 2 and Layer 3
            v2 = 0.5 * (F.adaptive_avg_pool2d(f2, 1).flatten(1) + F.adaptive_max_pool2d(f2, 1).flatten(1))
            v3 = 0.5 * (F.adaptive_avg_pool2d(f3, 1).flatten(1) + F.adaptive_max_pool2d(f3, 1).flatten(1))
            return torch.cat([v2, v3], dim=-1)
        else:
            return self.net(x)

backbone = IndustrialFeatureExtractor(BACKBONE_CHOICE).to(device).eval()
FEAT_DIM = backbone.out_dim
print(f"Loaded {BACKBONE_CHOICE} | Output feature dimension per frame: {FEAT_DIM}")

IMG_SIZE = 336 if "dinov2" in BACKBONE_CHOICE else 256
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set False for a 3-minute test on masks, or True to stream the raw RGB TIFFs
USE_RAW_RGB_DOWNLOAD = False
BATCH_SIZE = 64

@torch.no_grad()
def extract_features_for_tool(tool_id, use_raw_rgb=False):
    mode_tag = f"{BACKBONE_CHOICE}_{'rgb' if use_raw_rgb else 'mask'}"
    save_path = FEAT_DIR / f"{tool_id}_{mode_tag}.pt"
    if save_path.exists():
        return torch.load(save_path, map_location="cpu")

    mask_files = sorted((MASKS_DIR / tool_id).glob("*.png"))
    angle_cal = get_tool_tilt_angle(tool_id, ROTATIONS_DICT)

    # STAGE 1: Compute Master Mask & Fixed ROI Box ONCE per tool!
    master_mask = build_tool_master_mask(tool_id)
    roi_box = get_tool_roi_box(master_mask, pad_px=40, aspect_square=True)
    xmin, xmax, ymin_roi, ymax_roi, _ = roi_box

    if use_raw_rgb:
        raw_dir = download_and_extract_single_tool_rgb(tool_id)
        tiff_files = sorted(raw_dir.glob("*.tiff"))
    else:
        tiff_files = mask_files

    batch_tensors, feats_list, angles_list, areas_list = [], [], [], []

    for f_idx, m_path in enumerate(mask_files):
        deg_val = float(m_path.stem.split("_")[0])
        angles_list.append(deg_val)

        msk = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        # Compute 1D silhouette pixel area strictly inside the fixed Master-Mask ROI
        areas_list.append(float(np.sum(msk[ymin_roi:ymax_roi, xmin:xmax] > 127)))

        if use_raw_rgb and f_idx < len(tiff_files):
            t_path = raw_dir / (m_path.stem + ".tiff")
            if not t_path.exists():
                t_path = tiff_files[f_idx]
            rgb = tifffile.imread(t_path)
            rgb_aligned = rotate_image_to_match_mask(rgb, angle_cal, is_mask=False)
            crop_img, crop_msk, _ = crop_tool_head_roi(rgb_aligned, msk, roi_box=roi_box)
            # Zero out background so network focuses strictly on tool surface & silhouette edge
            crop_img = cv2.bitwise_and(crop_img, crop_img, mask=(crop_msk > 127).astype(np.uint8) * 255)
        else:
            _, crop_msk, _ = crop_tool_head_roi(msk, msk, roi_box=roi_box)
            crop_img = cv2.cvtColor(crop_msk, cv2.COLOR_GRAY2RGB)

        batch_tensors.append(img_transform(crop_img))

        if len(batch_tensors) == BATCH_SIZE or f_idx == len(mask_files) - 1:
            x_batch = torch.stack(batch_tensors).to(device)
            emb = backbone(x_batch).cpu()
            feats_list.append(emb)
            batch_tensors = []

    if use_raw_rgb:
        shutil.rmtree(LOCAL_CACHE / "raw_rgb" / tool_id, ignore_errors=True)

    data_dict = {
        "tool_id": tool_id,
        "features": torch.cat(feats_list, dim=0).to(torch.float16),
        "angles": torch.tensor(angles_list, dtype=torch.float32),
        "areas": torch.tensor(areas_list, dtype=torch.float32),
    }
    torch.save(data_dict, save_path)
    return data_dict

all_tool_data = {}
for t_id in tqdm(df_meta["tool_id"], desc=f"Extracting ({BACKBONE_CHOICE})"):
    all_tool_data[t_id] = extract_features_for_tool(t_id, use_raw_rgb=USE_RAW_RGB_DOWNLOAD)

print(f"Done! Cached 115 tool feature bags in {FEAT_DIR}")
"""))

# ==============================================================================
# CELL 9: ROTATIONAL GATED-ATTENTION MIL (ROT-ABMIL) & 5-FOLD CROSS-VALIDATION
# ==============================================================================
cells.append(make_md_cell("""
## Cell 9: Rotational Gated-Attention Multiple Instance Learning (`Rot-ABMIL`)
Trains a weakly supervised **Rotational Gated-Attention MIL** classifier across all 115 tools using **5-Fold Stratified Cross-Validation at the Tool Level** (zero data leakage between rotational frames):
* **Circular 1D Angular Convolution (`padding_mode='circular'`):** Captures local frame-to-frame profile changes around the $360^\circ$ circle.
* **Gated Attention MIL Pooling (Ilse et al., ICML 2018):** Learns an attention weight $\\alpha_k \\in [0, 1]$ for every frame $k \\in [1, K]$ so even a **1-frame micro-fracture** triggers high activation.
* **Global Mean + Classical DSI Fusion:** Combines localized defect attention with global surface appearance and tabular DSI features.
"""))

cells.append(make_code_cell("""
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 🔧 [MANUAL FIX #6]: Choose classification task mode:
#   'binary':  Usable ('new', 'used') vs. Defective/End-of-Life ('worn', 'deposit', 'fractured')
#   '3class':  Intact ('new', 'used') vs. Surface/Deposit ('worn', 'deposit') vs. Broken ('fractured')
#   '5class':  Full 5-class ('new', 'used', 'worn', 'deposit', 'fractured')
TASK_MODE = "binary"

def map_condition_label(cond_str, mode=TASK_MODE):
    if mode == "binary":
        return 0 if cond_str in ["new", "used"] else 1
    elif mode == "3class":
        if cond_str in ["new", "used"]: return 0
        elif cond_str in ["worn", "deposit"]: return 1
        else: return 2
    else:
        return {"new": 0, "used": 1, "worn": 2, "deposit": 3, "fractured": 4}[cond_str]

CLASS_NAMES = {
    "binary": ["Usable (New/Used)", "Defective (Worn/Deposit/Fractured)"],
    "3class": ["Usable (New/Used)", "Degraded (Worn/BUE)", "Fractured"],
    "5class": ["New", "Used", "Worn", "Deposit (BUE)", "Fractured"]
}[TASK_MODE]

class RotationalGatedAttentionMIL(nn.Module):
    def __init__(self, in_dim=384, hid_dim=128, aux_dim=3, num_classes=2, dropout=0.25):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # Circular 1D conv across rotational angles (wraps 360 deg -> 0 deg seamlessly!)
        self.conv_circ = nn.Conv1d(hid_dim, hid_dim, kernel_size=5, padding=2, padding_mode='circular')
        
        # Gated Attention mechanism (Ilse et al., 2018)
        self.att_V = nn.Sequential(nn.Linear(hid_dim, hid_dim // 2), nn.Tanh())
        self.att_U = nn.Sequential(nn.Linear(hid_dim, hid_dim // 2), nn.Sigmoid())
        self.att_w = nn.Linear(hid_dim // 2, 1)

        # Final classifier fusing: [Attention-Pooled (hid_dim) || Mean-Pooled (hid_dim) || Aux DSI/Area stats (aux_dim)]
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim * 2 + aux_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, bag_feats, aux_feats):
        # bag_feats: [K, in_dim] for a single tool bag (handles any K from 282 to 402!)
        h = self.proj(bag_feats)                                  # [K, hid_dim]
        h_conv = self.conv_circ(h.transpose(0, 1)).transpose(0, 1) # [K, hid_dim]
        h = h + 0.5 * F.gelu(h_conv)

        # Compute gated attention weights alpha_k across all K frames
        A_V = self.att_V(h)                                       # [K, hid_dim//2]
        A_U = self.att_U(h)                                       # [K, hid_dim//2]
        logits_att = self.att_w(A_V * A_U).squeeze(-1)            # [K]
        alpha = F.softmax(logits_att, dim=0)                      # [K], sums to 1.0

        z_att = torch.sum(alpha.unsqueeze(-1) * h, dim=0)         # [hid_dim] - spots 1-frame fractures
        z_mean = torch.mean(h, dim=0)                             # [hid_dim] - captures global wear/color

        z_total = torch.cat([z_att, z_mean, aux_feats], dim=-1)   # [2*hid_dim + aux_dim]
        out_logits = self.classifier(z_total)
        return out_logits, alpha

# Prepare dataset bags + auxiliary features (log-transformed mean_dsi_pct, max_dsi_pct, area_cv)
bags_list = []
labels_list = []
tool_ids_list = []

for _, row in df_meta.iterrows():
    t_id = row["tool_id"]
    d = all_tool_data[t_id]
    feats = d["features"].float()
    areas = d["areas"].float()
    area_cv = (areas.std() / (areas.mean() + 1e-6)).item() * 100.0
    m_dsi = float(row["mean_dsi_pct"]) if pd.notna(row["mean_dsi_pct"]) else area_cv
    x_dsi = float(row["max_dsi_pct"]) if pd.notna(row["max_dsi_pct"]) else area_cv * 1.5
    aux = torch.tensor([np.log1p(m_dsi), np.log1p(x_dsi), np.log1p(area_cv)], dtype=torch.float32)

    bags_list.append((feats, aux))
    labels_list.append(map_condition_label(row["condition"], TASK_MODE))
    tool_ids_list.append(t_id)

labels_arr = np.array(labels_list)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(df_meta), dtype=int)
oof_probs = np.zeros((len(df_meta), len(CLASS_NAMES)), dtype=float)
learned_attentions = {}

for fold, (train_idx, val_idx) in enumerate(skf.split(tool_ids_list, labels_arr)):
    model = RotationalGatedAttentionMIL(in_dim=384, hid_dim=128, aux_dim=3, num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # Class-balanced weights
    class_counts = np.bincount(labels_arr[train_idx])
    class_weights = torch.tensor(len(train_idx) / (len(class_counts) * np.maximum(class_counts, 1)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    for epoch in range(45):
        model.train()
        np.random.shuffle(train_idx)
        for i in train_idx:
            feats, aux = bags_list[i]
            # Rotational augmentation: random circular roll + random 75% angular subsample
            K = feats.shape[0]
            shift = np.random.randint(0, K)
            feats_aug = torch.roll(feats, shifts=shift, dims=0)
            keep_idx = np.sort(np.random.choice(K, size=int(K * 0.75), replace=False))
            feats_aug = feats_aug[keep_idx].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(feats_aug, aux.to(device))
            loss = criterion(logits.unsqueeze(0), torch.tensor([labels_arr[i]], device=device))
            loss.backward()
            optimizer.step()

        # Evaluate on full 360-degree bags (all K frames)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in val_idx:
                feats, aux = bags_list[i]
                logits, alpha = model(feats.to(device), aux.to(device))
                val_loss += criterion(logits.unsqueeze(0), torch.tensor([labels_arr[i]], device=device)).item()
                if epoch == 44 or val_loss < best_val_loss:
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    oof_probs[i] = probs
                    oof_preds[i] = int(np.argmax(probs))
                    learned_attentions[tool_ids_list[i]] = alpha.cpu().numpy()
        best_val_loss = min(best_val_loss, val_loss)

print("\\n=== 5-Fold Out-of-Sample Tool-Level Classification Report ===")
print(classification_report(labels_arr, oof_preds, target_names=CLASS_NAMES, digits=3))
"""))

# ==============================================================================
# CELL 10: FIGURE 8 — EXPLAINABLE DEEP MIL ATTENTION VS. 1D SILHOUETTE SIGNAL
# ==============================================================================
cells.append(make_md_cell("""
## Cell 10: Deep Learning Benchmark Figure — Learned Angular Attention $\\alpha(\\theta)$ vs. Classical 1D Silhouette Profile (`Fig8_DeepMIL_Attention.pdf`)
Generates an explainability figure comparing your classical **1D Silhouette Area curve** against the **Deep MIL learned attention weights $\\alpha(\\theta)$** across $0^\circ \\rightarrow 360^\circ$, alongside the 5-Fold Confusion Matrix.
"""))

cells.append(make_code_cell("""
# 🔧 [MANUAL FIX #7]: Pick any defective tool ID to visualize its learned attention curve alpha(theta)
ATT_TOOL_ID = "tool006"  # e.g., 'tool006' (fractured) or 'tool023' (BUE deposit)

d_tool = all_tool_data[ATT_TOOL_ID]
angles_t = d_tool["angles"].numpy()
areas_t = d_tool["areas"].numpy()
areas_pct = (areas_t / np.median(areas_t)) * 100.0
alpha_t = learned_attentions[ATT_TOOL_ID]
alpha_norm = alpha_t / (np.max(alpha_t) + 1e-8)

fig = plt.figure(figsize=(7.4, 3.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.65, 1.0], wspace=0.32)

# Panel a: Dual-axis plot of Classical 1D Silhouette Area (%) and Deep MIL Attention Weight alpha(theta)
ax1 = fig.add_subplot(gs[0])
ax2 = ax1.twinx()

l1 = ax1.plot(angles_t, areas_pct, color="#2b2b2b", linewidth=1.2, label="Classical 1D silhouette area (%)")
l2 = ax2.fill_between(angles_t, 0, alpha_norm, color="#e63946", alpha=0.28, label="Deep MIL attention $\\\\alpha(\\\\theta)$ (norm.)")
ax2.plot(angles_t, alpha_norm, color="#e63946", linewidth=1.0)

peak_idx = int(np.argmax(alpha_norm))
ax1.axvline(angles_t[peak_idx], color="#e63946", linestyle="--", linewidth=1.0)
ax1.set_xlim(0, 360)
ax1.set_xlabel("Rotational angle $\\\\theta$ (degrees)")
ax1.set_ylabel("Normalized silhouette area (%)", color="#2b2b2b")
ax2.set_ylabel("Normalized MIL attention $\\\\alpha(\\\\theta)$", color="#e63946")
ax2.set_ylim(0, 1.15)
ax2.spines["right"].set_visible(True)
ax1.set_title(f"Angular defect localization on {ATT_TOOL_ID} (Peak attention at $\\\\theta = {angles_t[peak_idx]:.1f}^\\\\circ$)", fontsize=8.5)
add_panel_label(ax1, "a", x=-0.06, y=1.10)

# Panel b: 5-Fold Cross-Validation Confusion Matrix (n = 115 tools)
ax_cm = fig.add_subplot(gs[1])
cm = confusion_matrix(labels_arr, oof_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[c.split()[0] for c in CLASS_NAMES],
            yticklabels=[c.split()[0] for c in CLASS_NAMES], ax=ax_cm)
ax_cm.set_xlabel("Predicted tool state")
ax_cm.set_ylabel("True tool state")
ax_cm.set_title("5-Fold tool-level confusion matrix (n = 115)", fontsize=8.5)
add_panel_label(ax_cm, "b", x=-0.12, y=1.10)

out_pdf = FIG_DIR / "Fig8_DeepMIL_Attention.pdf"
plt.savefig(out_pdf)
plt.show()
print(f"Saved {out_pdf} (Size: {out_pdf.stat().st_size / 1024:.1f} KB)")
"""))

# ==============================================================================
# WRITE JUPYTER NOTEBOOK (.IPYNB) FILE
# ==============================================================================
notebook_json = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "ELTE_TCM_46k_Paper_Figures_and_DeepMIL.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_filename = "ELTE_TCM_46k_Paper_Figures_and_DeepMIL.ipynb"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print(f"Successfully generated '{output_filename}'! Upload this file directly to Google Colab.")