#!/usr/bin/env python3
"""
Generate per-day PNG images combining ground-truth active fire masks and
model predictions for timeline figures.

Inputs expected under `hdf5/`:
- hdf5/true_data/<fire>.hdf5           (dataset: "data")
- hdf5/transformers/<fire>.hdf5        (dataset: one of {"pred_prob","data"})
- hdf5/UTAE_PhysicsLoss/<fire>.hdf5    (dataset: one of {"pred_prob","data"})

Output:
- A directory with two images per fire and per day, named by the date from
    HDF5 `img_dates`, suffixed per model. Each image overlays:
        - Ground truth: white (binary mask where value>0)
        - One model prediction as semi-transparent color
            - transformers: yellow
            - UTAE_PhysicsLoss: orange

If only a subset of inputs is available, the script will render what it finds.

Usage:
    python scripts/make_fire_timeline_images.py --hdf5-root hdf5 --out-dir timeline_images \
        --threshold 0.1 --pred-alpha 0.35
"""
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import h5py
from PIL import Image, ImageDraw, ImageFont


def load_true_data(true_path: Path, active_ch: int = -1) -> Tuple[List[np.ndarray], List[str]]:
    """Load true active-fire masks and dates.

    Returns a list of arrays (H, W) per time step, and a list of date strings.
    """
    with h5py.File(str(true_path), "r") as f:
        if "data" not in f:
            raise RuntimeError(f"True data file {true_path} missing dataset 'data'.")
        ds = f["data"][...]  # (T, C, H, W)
        img_dates = [d.decode("utf-8") if isinstance(d, (bytes, np.bytes_)) else str(d)
                     for d in f["data"].attrs.get("img_dates", [])]
    if ds.ndim != 4:
        raise RuntimeError(f"Unexpected shape {ds.shape} in {true_path} (expected T,C,H,W)")
    T, C, H, W = ds.shape
    ch = active_ch if active_ch >= 0 else C + active_ch
    true_masks = [(ds[t, ch] > 0).astype(np.uint8) for t in range(T)]
    return true_masks, img_dates


def load_prediction_frames(pred_path: Path) -> Optional[List[np.ndarray]]:
    """Load prediction frames as float arrays (H, W).

    Accepts shapes (T,H,W) or (T,C,H,W). Will prefer 'pred_prob' if present,
    else 'data', else first dataset.
    """
    if not pred_path.exists():
        return None
    try:
        with h5py.File(str(pred_path), "r") as f:
            if "pred_prob" in f:
                arr = f["pred_prob"][...]
            elif "data" in f:
                arr = f["data"][...]
            else:
                # pick first dataset key
                key = next(iter(f.keys()))
                arr = f[key][...]
        if arr.ndim == 3:
            T, H, W = arr.shape
            return [arr[t].astype(np.float32) for t in range(T)]
        elif arr.ndim == 4:
            T, C, H, W = arr.shape
            return [arr[t, -1].astype(np.float32) for t in range(T)]
        else:
            print(f"[!] Unexpected prediction shape {arr.shape} in {pred_path}")
            return None
    except Exception as e:
        print(f"[!] Failed reading predictions {pred_path}: {e}")
        return None


def make_overlay_rgba(base_size: Tuple[int, int], mask: np.ndarray, color: Tuple[int, int, int], alpha_scale: float) -> Image.Image:
    """Create an RGBA overlay where mask controls alpha.

    - base_size: (width, height)
    - mask: float32 array (H, W) normalized in [0,1]
    - color: (R,G,B)
    - alpha_scale: multiply mask to get alpha 0..255
    """
    h, w = mask.shape
    # resize mask to base size if needed
    if (w, h) != base_size:
        # PIL expects (width, height)
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        pil_mask = pil_mask.resize(base_size, resample=Image.NEAREST)
        mask = np.array(pil_mask).astype(np.float32) / 255.0
        h = base_size[1]
        w = base_size[0]
    alpha = np.clip(mask * (alpha_scale * 255.0), 0, 255).astype(np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def render_day_image(true_mask: Optional[np.ndarray],
                     pred: Optional[np.ndarray],
                     model_name: str,
                     date_str: str,
                     out_path: Path,
                     pred_threshold: float = 0.1,
                     pred_alpha: float = 0.35) -> None:
    """Render a single day image combining ground-truth and a single model overlay."""
    # Base canvas from true mask size or any available pred
    base_h = base_w = None
    if true_mask is not None:
        base_h, base_w = true_mask.shape
    elif pred is not None:
        base_h, base_w = pred.shape
    else:
        print(f"[!] No data available for {date_str}, skipping")
        return

    # Base image: white where true_mask is 1, black elsewhere
    if true_mask is None:
        base = Image.new("RGB", (base_w, base_h), (0, 0, 0))
    else:
        # ensure binary uint8
        tm = (true_mask > 0).astype(np.uint8)
        rgb = np.stack([tm * 255] * 3, axis=-1)
        base = Image.fromarray(rgb, mode="RGB")

    # Overlays: transformers (yellow), utae (orange)
    base = base.convert("RGBA")

    def norm_and_threshold(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        vmax = float(np.max(arr))
        if vmax <= 0:
            return None
        norm = arr / (vmax + 1e-12)
        # Dynamic thresholding to avoid full-screen activation
        p95 = float(np.percentile(norm, 95))
        thr = max(pred_threshold, min(0.99, p95))
        mask = (norm >= thr).astype(np.float32)
        return mask

    pred_norm = norm_and_threshold(pred)

    # Choose color per model
    color_map = {
        "transformers": (255, 255, 0),
        "UTAE_PhysicsLoss": (255, 165, 0),
    }
    color = color_map.get(model_name, (0, 255, 255))
    if pred_norm is not None:
        # Use solid color alpha (255) for strict palette
        overlay = make_overlay_rgba(base.size, pred_norm, color=color, alpha_scale=1.0)
        # If true mask is present, zero overlay alpha where true is active (keep colors strictly white/orange/black)
        if true_mask is not None:
            # Resize true to base size and binarize
            tm = (true_mask > 0).astype(np.uint8)
            tm_img = Image.fromarray(tm * 255)
            tm_img = tm_img.resize(base.size, resample=Image.NEAREST)
            tm_resized = (np.array(tm_img) > 0).astype(np.uint8)
            ov_arr = np.array(overlay)
            ov_arr[..., 3] = ov_arr[..., 3] * (1 - tm_resized)
            overlay = Image.fromarray(ov_arr, mode="RGBA")
        base = Image.alpha_composite(base, overlay)

    img = base.convert("RGB")

    # No labels on the image per request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG")


def find_prediction_path(root: Path, subdir: str, stem: str) -> Optional[Path]:
    """Try to locate a prediction file by stem under subdir."""
    sdir = root / subdir
    if not sdir.exists():
        return None
    # try exact name
    p = sdir / f"{stem}.hdf5"
    if p.exists():
        return p
    # fallback: glob any match
    matches = list(sdir.glob(f"{stem}*.hdf5")) + list(sdir.glob(f"{stem}*.h5"))
    return matches[0] if matches else None


def load_all_predictions_by_date(root: Path, subdir: str, fire_stem: str) -> dict:
    """Load all prediction frames from all HDF5s under a subdir, keyed by date string.

    If a file doesn't provide `img_dates`, we still store by index with keys like
    `t000`, `t001`, etc., so index-based alignment can work as a fallback.
    When multiple files provide the same key, prefer the one whose filename stem
    contains `fire_stem`.
    """
    sdir = root / subdir
    if not sdir.exists():
        return {}
    date_map = {}
    files = sorted(list(sdir.glob("*.hdf5")) + list(sdir.glob("*.h5")))
    for p in files:
        try:
            with h5py.File(str(p), "r") as f:
                # choose dataset
                if "pred_prob" in f:
                    arr = f["pred_prob"][...]
                elif "data" in f:
                    arr = f["data"][...]
                else:
                    key = next(iter(f.keys()))
                    arr = f[key][...]
                # dates
                ds_for_attr = f.get("pred_prob", f.get("data", None))
                has_dates = False
                img_dates = []
                if ds_for_attr is not None:
                    raw_dates = ds_for_attr.attrs.get("img_dates", [])
                    if len(raw_dates) > 0:
                        has_dates = True
                        img_dates = [d.decode("utf-8") if isinstance(d, (bytes, np.bytes_)) else str(d) for d in raw_dates]
                # reshape to (T,H,W) picking last channel if 4D
                if arr.ndim == 4:
                    T, C, H, W = arr.shape
                    frames = [arr[t, -1].astype(np.float32) for t in range(T)]
                elif arr.ndim == 3:
                    T, H, W = arr.shape
                    frames = [arr[t].astype(np.float32) for t in range(T)]
                else:
                    continue
                for t, frame in enumerate(frames):
                    date_key = img_dates[t] if (has_dates and t < len(img_dates)) else f"t{t:03d}"
                    # Prefer entries whose filename stem matches the fire stem; else fill if absent
                    if fire_stem in p.stem or date_key not in date_map:
                        date_map[date_key] = frame
        except Exception as e:
            print(f"[!] Failed reading {subdir} predictions {p}: {e}")
            continue
    return date_map


def main():
    ap = argparse.ArgumentParser(description="Generate per-day timeline images combining true data and model predictions.")
    ap.add_argument("--hdf5-root", default="hdf5", help="Root folder containing 'true_data', 'transformers', 'UTAE_PhysicsLoss'")
    ap.add_argument("--out-dir", default="timeline_images", help="Output root directory for generated PNGs")
    ap.add_argument("--active-ch", type=int, default=-1, help="Active fire channel in true_data (default -1 = last)")
    ap.add_argument("--threshold", type=float, default=0.1, help="Prediction threshold for binary mask (>= threshold => active)")
    ap.add_argument("--pred-alpha", type=float, default=1.0, help="Opacity for prediction overlays (0..1); set 1.0 for solid color")
    ap.add_argument("--true-offset", type=int, default=1, help="Offset for true_data alignment relative to predictions (e.g., 1 = use next day's active channel)")
    ap.add_argument("--fire", type=str, default=None, help="Process only a specific fire id or stem (e.g., 25294714 or full stem)")
    args = ap.parse_args()

    root = Path(args["hdf5_root"]) if isinstance(args, dict) else Path(args.hdf5_root)
    out_root = Path(args["out_dir"]) if isinstance(args, dict) else Path(args.out_dir)

    true_dir = root / "true_data"
    if not true_dir.exists():
        raise FileNotFoundError(f"Missing true_data directory under {root}")

    # Iterate true_data files as the timeline basis
    true_files = sorted(list(true_dir.glob("*.hdf5")) + list(true_dir.glob("*.h5")))
    # Optional filter by fire id/stem
    fire_filter = args.fire if not isinstance(args, dict) else args.get("fire")
    if fire_filter:
        fire_filter = str(fire_filter)
        true_files = [p for p in true_files if fire_filter in p.stem]
    if len(true_files) == 0:
        print(f"No true_data HDF5 files found under {true_dir}")
        return

    print(f"Found {len(true_files)} fires in true_data.")

    for true_path in true_files:
        fire_stem = true_path.stem
        print(f"[+] Processing fire {fire_stem}")
        try:
            true_masks, dates = load_true_data(true_path, active_ch=args.active_ch)
        except Exception as e:
            print(f"[!] Failed loading true data {true_path}: {e}")
            continue

        # Align predictions by date across all files in subdirs
        trans_by_date = load_all_predictions_by_date(root, "transformers", fire_stem)
        utae_by_date = load_all_predictions_by_date(root, "UTAE_PhysicsLoss", fire_stem)

        T = len(true_masks)
        for t in range(T):
            # Align true as next day (offset), default offset=1
            t_true = t + (args.true_offset if not isinstance(args, dict) else args["true_offset"])
            date_str = dates[t] if t < len(dates) else f"t{t:03d}"
            tm = true_masks[t_true] if (0 <= t_true < T) else None
            # Try date-based match first, then index-based fallback
            tp = trans_by_date.get(date_str, None)
            up = utae_by_date.get(date_str, None)
            if tp is None:
                tp = trans_by_date.get(f"t{t:03d}", None)
            if up is None:
                up = utae_by_date.get(f"t{t:03d}", None)

            # Save two images per day: true+UTAE and true+transformers
            try:
                if up is not None:
                    out_u = out_root / "UTAE_PhysicsLoss" / fire_stem / f"{date_str}.png"
                    render_day_image(
                        true_mask=tm,
                        pred=up,
                        model_name="UTAE_PhysicsLoss",
                        date_str=date_str,
                        out_path=out_u,
                        pred_threshold=args.threshold,
                        pred_alpha=args.pred_alpha,
                    )
                if tp is not None:
                    out_t = out_root / "transformers" / fire_stem / f"{date_str}.png"
                    render_day_image(
                        true_mask=tm,
                        pred=tp,
                        model_name="transformers",
                        date_str=date_str,
                        out_path=out_t,
                        pred_threshold=args.threshold,
                        pred_alpha=args.pred_alpha,
                    )
                if up is None and tp is None:
                    print(f"[!] No predictions for {fire_stem} {date_str}; only true mask available — skipping outputs.")
                if tm is None:
                    print(f"[!] No aligned true mask for {fire_stem} {date_str} (offset may exceed range). Rendering predictions over black background.")
            except Exception as e:
                print(f"[!] Error rendering {fire_stem} {date_str}: {e}")

    print(f"[✓] Timeline images written to {out_root}")


if __name__ == "__main__":
    main()
