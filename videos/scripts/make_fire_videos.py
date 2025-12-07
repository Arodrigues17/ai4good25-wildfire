#!/usr/bin/env python3
"""
Generate consistent-color MP4 videos directly from active-fire masks
stored in per-fire HDF5 files (dataset: "data").

For each HDF5 file:
    - Load data: (time, channels, H, W)
    - Extract active fire channel (default: last channel)
    - Load real dates from HDF5 attribute "img_dates"
    - Build a video with:
          t = X h (YYYY-MM-DD)
    - Fixed color scale across frames
    - Binary or continuous mode

Usage:
    python make_fire_videos.py /path/to/hdf5_folder --out-dir videos
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
import imageio.v2 as imageio
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont


# -------------------------------------------------------------------------
# Create video frames with timestamp + date
# -------------------------------------------------------------------------
def make_video_from_arrays(arr_list, out_path: Path, img_dates,
                           fps=6, cmap_name="magma",
                           vmin=None, vmax=None,
                           timestep_hours=24.0, binary=False,
                           pred_arr_list=None, pred_alpha=0.35,
                           pred_threshold=0.1, pred_vmax=None):
    """
    Create a video from raw 2D arrays using a fixed color scale,
    overlaying:  t = X h (YYYY-MM-DD)
    """

    # global min/max only for continuous values
    if not binary:
        if vmin is None:
            vmin = min(float(np.nanmin(a)) for a in arr_list)
        if vmax is None:
            vmax = max(float(np.nanmax(a)) for a in arr_list)
        cmap = cm.get_cmap(cmap_name)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264")

    # Load a readable font
    # We use a small size because fire tiles are small (128–512 px)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    for t, arr in enumerate(arr_list):

        # ---- Create RGB frame ----
        if binary:
            rgb = (arr.astype(np.uint8) * 255)
            rgb = np.stack([rgb] * 3, axis=-1)
        else:
            normed = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0, 1)
            rgb = (cmap(normed)[..., :3] * 255).astype(np.uint8)

        # ---- Convert to PIL for drawing text ----
        img = Image.fromarray(rgb)

                # ---- Overlay predictions (optional) ----
        if pred_arr_list is not None:
            try:
                pred = pred_arr_list[t]
            except Exception:
                pred = None

            if pred is not None:
                # Normalize prediction to 0..1
                pred = pred.astype(np.float32)
                if pred_vmax is None:
                    pmax = float(np.nanmax(pred)) if np.nanmax(pred) != 0 else 1.0
                else:
                    pmax = float(pred_vmax)
                pred_norm = np.clip(pred / (pmax + 1e-12), 0.0, 1.0)

                # Threshold to create a mask for overlay intensity
                mask = (pred_norm >= pred_threshold).astype(np.float32)

                # Alpha channel per-pixel from prediction strength
                alpha_arr = (pred_norm * mask * (pred_alpha * 255.0)).astype(np.uint8)

                # Build RGBA overlay (yellow)
                rgba = np.zeros((alpha_arr.shape[0], alpha_arr.shape[1], 4), dtype=np.uint8)
                rgba[..., 0] = 255  # R
                rgba[..., 1] = 255  # G
                rgba[..., 2] = 0    # B
                rgba[..., 3] = alpha_arr

                overlay = Image.fromarray(rgba, mode="RGBA")

                # --- Log and fix size mismatches by resizing overlay ---
                if pred.shape != arr.shape or overlay.size != img.size:
                    print(
                        f"[!] Resizing overlay at t={t}: "
                        f"prediction array shape {pred.shape} vs frame array shape {arr.shape}, "
                        f"overlay size {overlay.size} vs frame image size {img.size}. "
                        "Overlay will be resized to match the frame."
                    )
                    # img.size is (width, height)
                    overlay = overlay.resize(img.size, resample=Image.NEAREST)

                # Composite overlay on top of base image
                img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(img)

        # ---- Date extraction ----
        date_raw = img_dates[t]
        date_str = date_raw.decode("utf-8") if isinstance(date_raw, (bytes, np.bytes_)) else str(date_raw)

        # ---- Time formatting ----
        hours = t * timestep_hours
        text = f"t = {hours:.0f} h ({date_str})"

        # ---- Measure text size ----
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # ---- Safe placement ----
        margin = 5
        x = margin
        y = margin

        # ---- Background box ----
        # Use an RGBA tuple when drawing onto RGBA images; PIL will handle it for RGB too
        draw.rectangle(
            [x - 2, y - 2, x + text_w + 2, y + text_h + 2],
            fill=(0, 0, 0, 120)   # semi-transparent black
        )

        # ---- Draw text ----
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

        writer.append_data(np.array(img))

    writer.close()
    print(f"[✓] Saved video with timestamps: {out_path}")


# -------------------------------------------------------------------------
# Process one fire HDF5 → MP4 video
# -------------------------------------------------------------------------
def process_single_hdf5(hdf5_path: Path, out_vid: Path,
                        active_ch: int = -1, binary=False, fps=6,
                        pred_dir: Path = None, pred_alpha=0.35,
                        pred_threshold=0.1, pred_vmax=None):

    print(f"[+] Processing {hdf5_path}")

    # Load full array + date strings
    with h5py.File(str(hdf5_path), "r") as f:
        data = f["data"][...]               # shape (T, C, H, W)
        img_dates = f["data"].attrs["img_dates"]

    T, C, H, W = data.shape

    # Resolve channel index
    ch = active_ch if active_ch >= 0 else C + active_ch

    # Extract raw frames
    frames = []
    for t in range(T):
        arr = data[t, ch]
        if binary:
            arr = (arr > 0).astype(np.uint8)
        frames.append(arr)

    # Optionally attempt to load prediction file with the same filename
    pred_frames = None
    if pred_dir is not None:
        pred_dirp = Path(pred_dir)
        # Try to find file with same name under pred_dir
        matches = list(pred_dirp.rglob(hdf5_path.name))
        if len(matches) == 0:
            # also try matching by stem
            matches = list(pred_dirp.rglob(hdf5_path.stem + "*.h5")) + list(pred_dirp.rglob(hdf5_path.stem + "*.hdf5"))

        if len(matches) > 0:
            pred_path = matches[0]
            print(f"[i] Found predictions for {hdf5_path.stem}: {pred_path}")
            try:
                with h5py.File(str(pred_path), "r") as pf:
                    # prefer common dataset names
                    if "predictions" in pf:
                        ds = pf["predictions"]
                    elif "data" in pf:
                        ds = pf["data"]
                    else:
                        # pick first dataset-like entry
                        key = next(iter(pf.keys()))
                        ds = pf[key]

                    pred_data = ds[...]

                # Normalize shape: allow (T, H, W) or (T, C, H, W)
                if pred_data.ndim == 4:
                    # pick last channel
                    pT, pC, pH, pW = pred_data.shape
                    pred_ch = -1
                    pred_frames = [pred_data[t, pred_ch] for t in range(min(pT, T))]
                elif pred_data.ndim == 3:
                    pT, pH, pW = pred_data.shape
                    pred_frames = [pred_data[t] for t in range(min(pT, T))]
                else:
                    print(f"[!] Unexpected prediction data shape: {pred_data.shape}")
                    pred_frames = None

                # If length mismatch, pad with zeros or truncate
                if pred_frames is not None and len(pred_frames) < T:
                    # pad remaining with zeros
                    h, w = pred_frames[0].shape
                    for _ in range(T - len(pred_frames)):
                        pred_frames.append(np.zeros((h, w), dtype=np.float32))

            except Exception as e:
                print(f"[!] Could not load predictions from {pred_path}: {e}")
                pred_frames = None
        else:
            print(f"[i] No prediction file found for {hdf5_path.stem} under {pred_dir}")

    # Build video (24 h per frame)
    make_video_from_arrays(
        frames,
        out_vid,
        img_dates=img_dates,
        fps=fps,
        binary=binary,
        timestep_hours=24.0,
        pred_arr_list=pred_frames,
        pred_alpha=pred_alpha,
        pred_threshold=pred_threshold,
        pred_vmax=pred_vmax
    )


# -------------------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Create fire videos from HDF5 files with real dates & elapsed hours.")
    ap.add_argument("data_dir", help="Directory containing .h5/.hdf5 fire files")
    ap.add_argument("--out-dir", default="fire_videos", help="Where to save MP4s")
    ap.add_argument("--active-ch", type=int, default=-1, help="Active fire channel index (default: -1 = last)")
    ap.add_argument("--binary", action="store_true", default=True, help="Use binary masks instead of continuous values")
    ap.add_argument("--fps", type=int, default=6, help="Video framerate")
    ap.add_argument("--pred-dir", default=None, help="Directory containing prediction HDF5 files (optional)")
    ap.add_argument("--pred-alpha", type=float, default=0.35, help="Alpha for prediction overlay (0..1)")
    ap.add_argument("--pred-threshold", type=float, default=0.1, help="Threshold for prediction overlay (0..1)")
    ap.add_argument("--pred-vmax", type=float, default=None, help="Optional vmax to normalize predictions")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all fires
    files = sorted(list(data_dir.rglob("*.h5")) + list(data_dir.rglob("*.hdf5")))
    if len(files) == 0:
        print("No HDF5 files found.")
        return

    print(f"Found {len(files)} HDF5 fires.")

    # Process each fire into a video
    for f in files:
        out_vid = out_dir / f"{f.stem}.mp4"

        process_single_hdf5(
            f,
            out_vid,
            active_ch=args.active_ch,
            binary=args.binary,
            fps=args.fps,
            pred_dir=args.pred_dir,
            pred_alpha=args.pred_alpha,
            pred_threshold=args.pred_threshold,
            pred_vmax=args.pred_vmax
        )


if __name__ == "__main__":
    main()
