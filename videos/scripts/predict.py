#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader.FireSpreadDataset import FireSpreadDataset
from src.dataloader.FireSpreadDataModule import FireSpreadDataModule
from src.models import SMPModel, ConvLSTMLightning, LogisticRegression, UTAEContinuous, UTAELightning  # type: ignore


def discover_years_in_hdf5_root(data_dir: str) -> List[int]:
    years: List[int] = []
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    for child in p.iterdir():
        if child.is_dir():
            try:
                year = int(child.name)
                # Only add years that actually contain .hdf5 files
                if any(child.glob("*.hdf5")):
                    years.append(year)
            except ValueError:
                continue
    if not years:
        # Fallback: search recursively for .hdf5 and infer years from parent dir names
        for fp in p.rglob("*.hdf5"):
            try:
                years.append(int(fp.parent.name))
            except ValueError:
                pass
        years = sorted(list(set(years)))
    years.sort()
    if not years:
        raise RuntimeError(f"No .hdf5 fires found under {data_dir}")
    return years


def build_dataset(
    data_dir: str,
    included_years: List[int],
    n_leading_observations: int,
    crop_side_length: int,
    remove_duplicate_features: bool,
    features_to_keep: Optional[List[int]],
    stats_years: List[int],
    use_doy: bool,
) -> FireSpreadDataset:
    ds = FireSpreadDataset(
        data_dir=data_dir,
        included_fire_years=included_years,
        n_leading_observations=n_leading_observations,
        n_leading_observations_test_adjustment=None,
        crop_side_length=crop_side_length,
        load_from_hdf5=True,
        is_train=False,
        remove_duplicate_features=remove_duplicate_features,
        stats_years=stats_years,
        features_to_keep=features_to_keep,
        return_doy=use_doy,
        use_gaussian_targets=False,
    )
    return ds


def map_model_class(name: str):
    name = name.lower()
    if name in ("smp", "smpmodel"):
        return SMPModel
    if name in ("convlstm", "convlstm_lightning", "convlstmlightning"):
        return ConvLSTMLightning
    if name in ("logreg", "logistic", "logisticregression"):
        return LogisticRegression
    if name in ("utae", "utaelightning"):
        return UTAELightning
    if name in ("utaecontinuous", "utae_continuous"):
        return UTAEContinuous
    raise ValueError(
        f"Unknown model '{name}'. Choose from: smp, convlstm, logisticregression, utae, utaecontinuous"
    )


def ensure_outfile(
    out_path: Path,
    n_samples: int,
    height: int,
    width: int,
    target_dates: List[str],
    overwrite: bool,
    save_inputs: bool,
    save_targets: bool,
) -> h5py.File:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite:
        out_path.unlink()
    f = h5py.File(out_path, "w")
    f.create_dataset("pred_prob", shape=(n_samples, height, width), dtype=np.float32)
    f.create_dataset("pred_logit", shape=(n_samples, height, width), dtype=np.float32)
    dt = h5py.string_dtype(encoding="utf-8")
    f.create_dataset("target_dates", data=np.array(target_dates, dtype=dt), dtype=dt)
    if save_targets:
        f.create_dataset("target", shape=(n_samples, height, width), dtype=np.uint8)
    if save_inputs:
        f.create_dataset("input_active_fire", shape=(n_samples, height, width), dtype=np.uint8)
    return f


def main():
    parser = argparse.ArgumentParser(description="Run wildfire predictions over all HDF5 inputs and store outputs per fire.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root folder with year subfolders containing .hdf5 files")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--model", type=str, required=True, help="Model class: smp | convlstm | logisticregression | utae | utaecontinuous")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store per-fire HDF5 prediction files")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size. Keep 1 for variable-size inference")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_leading_observations", type=int, default=5)
    parser.add_argument("--crop_side_length", type=int, default=128)
    parser.add_argument("--remove_duplicate_features", action="store_true")
    parser.add_argument("--features_to_keep", type=str, default="", help="Comma-separated list of feature indices to keep. Empty=all")
    parser.add_argument("--data_fold_id", type=int, default=0, help="Fold id for computing normalization stats (defines training years)")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    parser.add_argument("--years", type=str, default="", help="Comma-separated list of years to run. Empty=auto-discover")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--save_inputs", action="store_true", help="Also store last-day input active fire mask")
    parser.add_argument("--save_targets", action="store_true", help="Also store target ground-truth mask")

    args = parser.parse_args()

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Years
    if args.years.strip():
        included_years = sorted(list({int(y) for y in args.years.split(",")}))
    else:
        included_years = discover_years_in_hdf5_root(args.data_dir)

    # Stats from training years according to fold
    train_years, _, _ = FireSpreadDataModule.split_fires(args.data_fold_id)

    # features_to_keep parsing
    features_to_keep: Optional[List[int]]
    if args.features_to_keep.strip():
        features_to_keep = sorted(list({int(i) for i in args.features_to_keep.split(",")}))
    else:
        features_to_keep = None

    # Build dataset and loader (return_doy follows model requirement)
    # We'll load the model first to know if it uses DOY
    ModelClass = map_model_class(args.model)
    # Load on CPU first to avoid device mismatch during init
    model = ModelClass.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()
    model.to(device)

    use_doy = bool(getattr(model.hparams, "use_doy", False))

    dataset = build_dataset(
        data_dir=args.data_dir,
        included_years=included_years,
        n_leading_observations=args.n_leading_observations,
        crop_side_length=args.crop_side_length,
        remove_duplicate_features=args.remove_duplicate_features,
        features_to_keep=features_to_keep,
        stats_years=train_years,
        use_doy=use_doy,
    )

    # For variable-size images, keep batch_size=1
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Pre-open input HDF5 files per fire, precompute target dates, and prepare out files lazily
    out_root = Path(args.output_dir)
    out_files: Dict[Tuple[int, str], h5py.File] = {}
    created_shapes: Dict[Tuple[int, str], Tuple[int, int]] = {}
    n_samples_per_fire: Dict[Tuple[int, str], int] = {}
    target_dates_per_fire: Dict[Tuple[int, str], List[str]] = {}

    # Helper to prepare file for a fire
    def prepare_fire_files(year: int, fire_name: str, sample_shape_hw: Tuple[int, int]):
        key = (year, fire_name)
        if key in out_files:
            return
        # samples for this fire
        n_samples = dataset.datapoints_per_fire[year][fire_name]
        n_samples_per_fire[key] = n_samples

        # read target dates from source HDF5
        hdf5_src_path = Path(dataset.imgs_per_fire[year][fire_name][0])
        with h5py.File(hdf5_src_path, "r") as fin:
            all_dates = [d for d in fin["data"].attrs["img_dates"]]
        # Ensure str list
        all_dates = [d.decode("utf-8") if isinstance(d, (bytes, np.bytes_)) else str(d) for d in all_dates]
        # target date is the (i + n_leading_observations)-th image
        target_dates = all_dates[args.n_leading_observations : args.n_leading_observations + n_samples]
        target_dates_per_fire[key] = target_dates

        # Create out file
        year_dir = out_root / str(year)
        out_path = year_dir / f"{fire_name}.hdf5"
        H, W = sample_shape_hw
        created_shapes[key] = (H, W)
        out_files[key] = ensure_outfile(
            out_path=out_path,
            n_samples=n_samples,
            height=H,
            width=W,
            target_dates=target_dates,
            overwrite=args.overwrite,
            save_inputs=args.save_inputs,
            save_targets=args.save_targets,
        )

    # Iterate samples
    global_index = 0
    with torch.no_grad():
        for batch in loader:
            # Determine fire id for this dataset index
            year, fire_name, in_fire_index = dataset.find_image_index_from_dataset_index(global_index)

            if use_doy:
                x, y, doys = batch
                x = x.to(device)
                y = y.to(device)
                doys = doys.to(device)
                y_hat, y_out = model.get_pred_and_gt((x, y, doys))
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y_hat, y_out = model.get_pred_and_gt((x, y))

            # y_hat shape: [B, H, W]; y_out shape: [B, H, W]
            y_hat = y_hat.detach().cpu().squeeze(0)
            y_out = y_out.detach().cpu().squeeze(0)
            prob = torch.sigmoid(y_hat)

            # Prepare output file lazily
            prepare_fire_files(year, fire_name, (y_hat.shape[-2], y_hat.shape[-1]))
            fkey = (year, fire_name)
            fout = out_files[fkey]

            fout["pred_prob"][in_fire_index, :, :] = prob.numpy().astype(np.float32)
            fout["pred_logit"][in_fire_index, :, :] = y_hat.numpy().astype(np.float32)

            if args.save_targets:
                fout["target"][in_fire_index, :, :] = (y_out.numpy() > 0).astype(np.uint8)

            if args.save_inputs:
                # best-effort: only works when temporal dimension preserved (no deduplication)
                x_cpu = x.detach().cpu().squeeze(0)
                if x_cpu.ndim == 4:  # [T, C, H, W]
                    last_af = (x_cpu[-1, -1, :, :] > 0).numpy().astype(np.uint8)
                    fout["input_active_fire"][in_fire_index, :, :] = last_af

            global_index += 1

    # Close files
    for f in out_files.values():
        try:
            f.flush()
            f.close()
        except Exception:
            pass

    print(f"Saved predictions for {len(out_files)} fires under: {args.output_dir}")


if __name__ == "__main__":
    main()
