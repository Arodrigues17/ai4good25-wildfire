#!/usr/bin/env python3
"""
Quick utility to visualize active fire masks from a dataset (HDF5 per-fire files or folders of TIFFs).

Features:
- Load a single HDF5 file (expected dataset name: "data") or a folder of TIFFs
- Show RGB composite (user-selectable channels), raw active-fire channel, binary mask, temporal union of detections
- Overlay masks on RGB for quick inspection

Usage examples:
  python scripts/plot_active_fire_masks.py /path/to/fire.hdf5
  python scripts/plot_active_fire_masks.py /path/to/fire_folder --rgb 0,1,2 --active-ch -1

Notes:
- The code assumes the HDF5 dataset stores data in the shape (time, channels, height, width),
  and that the active-fire channel is the last channel by default (index -1). This matches
  the conventions used in `src/dataloader/FireSpreadDataset.py`.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import h5py
except Exception:
    h5py = None

try:
    import rasterio
except Exception:
    rasterio = None


def load_hdf5(path: str):
    if h5py is None:
        raise RuntimeError("h5py is required to read HDF5 files. Install via `pip install h5py`.")
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"HDF5 file {path} does not contain dataset 'data'. Found: {list(f.keys())}")
        data = f["data"][...]
        attrs = dict(f["data"].attrs)
    return data, attrs


def load_tif_stack_from_folder(folder: str):
    if rasterio is None:
        raise RuntimeError("rasterio is required to read TIFF files. Install via `pip install rasterio`.")
    tifs = sorted([str(p) for p in Path(folder).glob("*.tif")])
    if len(tifs) == 0:
        raise RuntimeError(f"No .tif files found in folder {folder}")
    imgs = []
    for t in tifs:
        with rasterio.open(t) as ds:
            # ds.read() -> (channels, H, W)
            imgs.append(ds.read())
    # stack into (time, channels, H, W)
    return np.stack(imgs, axis=0), {"files": tifs}


def channel_to_rgb(arr_channels: np.ndarray, rgb_indices):
    # arr_channels shape (channels, H, W)
    H = arr_channels.shape[1]
    W = arr_channels.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(3):
        idx = rgb_indices[i]
        if idx is None or idx >= arr_channels.shape[0] or idx < -arr_channels.shape[0]:
            rgb[:, :, i] = 0.0
        else:
            ch = arr_channels[idx]
            # normalize channel for display
            p1, p99 = np.nanpercentile(ch, (1, 99))
            if p99 - p1 <= 0:
                rgb[:, :, i] = np.clip(ch, 0, 1)
            else:
                rgb[:, :, i] = np.clip((ch - p1) / (p99 - p1), 0, 1)
    return rgb


def plot_masks(data: np.ndarray, attrs: dict, args):
    # data shape expected (time, channels, H, W)
    if data.ndim != 4:
        raise RuntimeError(f"Expected data with 4 dims (time, channels, H, W), got {data.shape}")

    time_idx = args.time_index if args.time_index is not None else data.shape[0] - 1
    time_idx = int(time_idx)
    time_idx = np.clip(time_idx, 0, data.shape[0] - 1)

    channels = data.shape[1]
    # resolve active channel index
    aidx = args.active_ch
    if aidx < 0:
        aidx = channels + aidx

    selected = data[time_idx]  # (channels, H, W)

    # compute masks
    raw_active = selected[aidx]
    binary_active = (raw_active > 0).astype(np.uint8)

    # temporal union (any detection across time)
    all_active = (data[:, aidx, :, :] > 0).any(axis=0).astype(np.uint8)
    sum_active = (data[:, aidx, :, :] > 0).sum(axis=0)

    # RGB
    rgb_idxs = [int(x) if x is not None else None for x in args.rgb]
    rgb = channel_to_rgb(selected, rgb_idxs)

    # plotting
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()

    axs[0].set_title(f"RGB (time={time_idx})")
    axs[0].imshow(rgb)
    axs[0].axis("off")

    axs[1].set_title("Raw active-fire channel (float)")
    im1 = axs[1].imshow(raw_active, cmap="magma")
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    axs[2].set_title("Binary active-fire mask (time={})".format(time_idx))
    axs[2].imshow(binary_active, cmap="gray")
    axs[2].axis("off")

    axs[3].set_title("Temporal union of active detections")
    axs[3].imshow(all_active, cmap="gray")
    axs[3].axis("off")

    axs[4].set_title("Temporal sum of detections (counts)")
    im2 = axs[4].imshow(sum_active, cmap="viridis")
    axs[4].axis("off")
    fig.colorbar(im2, ax=axs[4], fraction=0.046, pad=0.04)

    # overlay union on RGB
    axs[5].set_title("RGB with union overlay")
    axs[5].imshow(rgb)
    # create red overlay where union is True
    mask = all_active.astype(bool)
    if mask.any():
        axs[5].contour(mask, colors="red", linewidths=0.5)
    axs[5].axis("off")

    plt.tight_layout()
    plt.savefig("active_fire_masks_plot.png")
    plt.show()


def process_hdf5_file(hdf5_path: Path, out_dir: Path, active_ch: int = -1, binary: bool = True, max_per_file=None):
    """Process a single HDF5 file: for each time index t, save a plot with input (t) on left and output (t+1) on right.

    Args:
        hdf5_path: path to the .h5/.hdf5 file
        out_dir: directory where plots will be saved
        active_ch: index of active-fire channel (negative allowed)
        binary: if True, show binary masks (value>0)
        max_per_file: optional int, stop after this many saved plots for the file
    """
    if h5py is None:
        raise RuntimeError("h5py is required to read HDF5 files. Install via `pip install h5py`.")

    with h5py.File(str(hdf5_path), "r") as f:
        if "data" not in f:
            raise RuntimeError(f"HDF5 file {hdf5_path} does not contain dataset 'data'.")
        ds = f["data"]
        # ds shape: (time, channels, H, W)
        if len(ds.shape) != 4:
            raise RuntimeError(f"Unexpected dataset shape {ds.shape} in {hdf5_path}")
        n_time, n_ch, H, W = ds.shape

        # resolve active channel index
        aidx = active_ch
        if aidx < 0:
            aidx = n_ch + aidx
        if aidx < 0 or aidx >= n_ch:
            raise RuntimeError(f"Active channel {active_ch} resolves to {aidx} which is out of bounds for {n_ch} channels")

        n_saved = 0
        for t in range(n_time - 1):
            in_ch = ds[t, aidx, :, :]
            out_ch = ds[t + 1, aidx, :, :]

            # convert to numpy arrays (h5py returns array-like)
            in_arr = np.array(in_ch)
            out_arr = np.array(out_ch)

            if binary:
                in_disp = (in_arr > 0).astype(np.uint8)
                out_disp = (out_arr > 0).astype(np.uint8)
                cmap = "gray"
            else:
                in_disp = np.nan_to_num(in_arr)
                out_disp = np.nan_to_num(out_arr)
                cmap = "magma"

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].set_title(f"Input t={t}")
            im0 = axs[0].imshow(in_disp, cmap=cmap)
            axs[0].axis("off")
            if not binary:
                fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            axs[1].set_title(f"Output t={t+1}")
            im1 = axs[1].imshow(out_disp, cmap=cmap)
            axs[1].axis("off")
            if not binary:
                fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            plt.suptitle(f"{hdf5_path.stem} index {t}")
            plt.tight_layout()

            out_file = out_dir / f"{hdf5_path.stem}_{t:05d}.png"
            fig.savefig(str(out_file), dpi=150)
            plt.close(fig)

            n_saved += 1
            if max_per_file is not None and n_saved >= max_per_file:
                break

    print(f"Saved {n_saved} plots for {hdf5_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Batch-plot active fire masks from a directory of HDF5 files")
    p.add_argument("data_dir", help="Root folder containing .h5/.hdf5 files (will search recursively)")
    p.add_argument("--out-dir", default="plots_active_fire", help="Directory to save generated plots")
    p.add_argument("--active-ch", type=int, default=-1, help="Index of active fire channel (default: -1, last channel)")
    p.add_argument("--binary", action="store_true", help="Save binary masks (detection>0). If not set, raw channel values are shown")
    p.add_argument("--max-per-file", type=int, default=None, help="Optional: limit number of plots saved per HDF5 file (for quick checks)")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    # find all .h5/.hdf5 files recursively
    hdf5_files = sorted(list(data_dir.rglob("*.h5")) + list(data_dir.rglob("*.hdf5")))
    if len(hdf5_files) == 0:
        print(f"No HDF5 files found under {data_dir}")
        return

    print(f"Found {len(hdf5_files)} HDF5 files. Processing...")
    for hdf5_path in hdf5_files:
        rel = hdf5_path.relative_to(data_dir)
        # create per-file output directory (preserve subdirs)
        file_out_dir = out_root / rel.parent / hdf5_path.stem
        file_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {hdf5_path} -> {file_out_dir}")
        try:
            process_hdf5_file(hdf5_path, file_out_dir, args.active_ch, args.binary, args.max_per_file)
        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")


if __name__ == "__main__":
    main()
