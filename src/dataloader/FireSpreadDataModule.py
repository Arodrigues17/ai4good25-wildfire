from pathlib import Path
import math
from functools import partial
import warnings

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import glob
from .FireSpreadDataset import FireSpreadDataset
from typing import List, Optional, Union
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple is None or multiple <= 1:
        return value
    return int(math.ceil(value / multiple) * multiple)


def _pad_spatial_tensor(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    pad_h = target_h - tensor.shape[-2]
    pad_w = target_w - tensor.shape[-1]
    if pad_h == 0 and pad_w == 0:
        return tensor.clone()
    orig_shape = tensor.shape
    lead_dims = orig_shape[:-2]
    flat = tensor.reshape(-1, orig_shape[-2], orig_shape[-1])
    padded = F.pad(flat, (0, pad_w, 0, pad_h), mode="reflect")
    new_h, new_w = padded.shape[-2:]
    return padded.reshape(*lead_dims, new_h, new_w)


def wildfire_collate(batch, pad_multiple: int = 1):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        has_spatial_dims = elem.dim() >= 2 and elem.shape[-2] > 4 and elem.shape[-1] > 4
        if has_spatial_dims:
            max_h = max(t.shape[-2] for t in batch)
            max_w = max(t.shape[-1] for t in batch)
            target_h = _round_up_to_multiple(max_h, pad_multiple)
            target_w = _round_up_to_multiple(max_w, pad_multiple)
            padded = [_pad_spatial_tensor(t, target_h, target_w) for t in batch]
            return torch.stack(padded, dim=0)
        return torch.stack([b.clone() for b in batch], dim=0)
    if isinstance(elem, dict):
        return {key: wildfire_collate([sample[key] for sample in batch], pad_multiple=pad_multiple) for key in elem}
    return default_collate(batch)


class FireSpreadDataModule(LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, n_leading_observations: int, n_leading_observations_test_adjustment: int,
                 crop_side_length: int,
                 load_from_hdf5: bool, num_workers: int, remove_duplicate_features: bool,
                 features_to_keep: Union[Optional[List[int]], str] = None, return_doy: bool = False,
                 return_metadata: bool = False, data_fold_id: int = 0, eval_pad_multiple: Optional[int] = 32,
                 standardize_features: bool = True, eval_batch_size: Optional[int] = None,
                 *args, **kwargs):
        """_summary_ Data module for loading the WildfireSpreadTS dataset.

        Args:
            data_dir (str): _description_ Path to the directory containing the data.
            batch_size (int): _description_ Batch size for training and validation set. Test set uses batch size 1, because images of different sizes can not be batched together.
            n_leading_observations (int): _description_ Number of days to use as input observation. 
            n_leading_observations_test_adjustment (int): _description_ When increasing the number of leading observations, the number of samples per fire is reduced.
              This parameter allows to adjust the number of samples in the test set to be the same across several different values of n_leading_observations, 
              by skipping some initial fires. For example, if this is set to 5, and n_leading_observations is set to 1, the first four samples that would be 
              in the test set are skipped. This way, the test set is the same as it would be for n_leading_observations=5, thereby retaining comparability 
              of the test set.
            crop_side_length (int): _description_ The side length of the random square crops that are computed during training and validation.
            load_from_hdf5 (bool): _description_ If True, load data from HDF5 files instead of TIF. 
            num_workers (int): _description_ Number of workers for the dataloader.
            remove_duplicate_features (bool): _description_ Remove duplicate static features from all time steps but the last one. Requires flattening the temporal dimension, since after removal, the number of features is not the same across time steps anymore.
            features_to_keep (Union[Optional[List[int]], str], optional): _description_. List of feature indices from 0 to 39, indicating which features to keep. Defaults to None, which means using all features.
            return_doy (bool, optional): _description_. Return the day of the year per time step, as an additional feature. Defaults to False.
            return_metadata (bool, optional): _description_. Return additional metadata (temporal coordinates and location) required by Prithvi models. Defaults to False.
            data_fold_id (int, optional): _description_. Which data fold to use, i.e. splitting years into train/val/test set. Defaults to 0.
            eval_pad_multiple (Optional[int], optional): _description_. Multiple to which validation/test samples are padded.
              Defaults to 32 to maintain previous behaviour; set to None to disable padding.
            eval_batch_size (Optional[int], optional): _description_. Batch size to use for evaluation/predict loaders.
              Defaults to None, which reuses `batch_size`.
        """
        super().__init__()

        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.data_fold_id = data_fold_id
        if not return_doy:
            warnings.warn(
                "Prithvi-based experiments rely on day-of-year features; overriding return_doy=True.",
                RuntimeWarning,
            )
        self.return_doy = True
        if not return_metadata:
            warnings.warn(
                "Prithvi-based experiments require temporal/location metadata; overriding return_metadata=True.",
                RuntimeWarning,
            )
        self.return_metadata = True
        # wandb apparently can't pass None values via the command line without turning them into a string, so we need this workaround
        self.features_to_keep = features_to_keep if type(
            features_to_keep) != str else None
        self.remove_duplicate_features = remove_duplicate_features
        self.num_workers = num_workers
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.eval_pad_multiple = eval_pad_multiple
        self.standardize_features = standardize_features
        self.eval_batch_size = eval_batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def _loader(self, dataset, *, shuffle, batch_size):
        pad_multiple = 1
        if self.eval_pad_multiple and self.eval_pad_multiple > 1:
            pad_multiple = int(self.eval_pad_multiple)
        collate = partial(wildfire_collate, pad_multiple=pad_multiple)
        kwargs = dict(batch_size=batch_size, shuffle=shuffle,
                      num_workers=self.num_workers, pin_memory=True,
                      collate_fn=collate)
        if self.num_workers:
            kwargs["multiprocessing_context"] = "spawn"
            kwargs["persistent_workers"] = True
        return DataLoader(dataset, **kwargs)

    def setup(self, stage: str):
        train_years, val_years, test_years = self.split_fires(
            self.data_fold_id)
        self.train_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=train_years,
                                               n_leading_observations=self.n_leading_observations,
                                               n_leading_observations_test_adjustment=None,
                                               crop_side_length=self.crop_side_length,
                                               load_from_hdf5=self.load_from_hdf5, is_train=True,
                                               remove_duplicate_features=self.remove_duplicate_features,
                                               features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                               return_metadata=self.return_metadata,
                                               eval_pad_multiple=self.eval_pad_multiple,
                                               standardize_features=self.standardize_features,
                                               stats_years=train_years)
        self.val_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=val_years,
                                             n_leading_observations=self.n_leading_observations,
                                             n_leading_observations_test_adjustment=None,
                                             crop_side_length=self.crop_side_length,
                                             load_from_hdf5=self.load_from_hdf5, is_train=False,
                                             remove_duplicate_features=self.remove_duplicate_features,
                                             features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                             return_metadata=self.return_metadata,
                                             eval_pad_multiple=self.eval_pad_multiple,
                                             standardize_features=self.standardize_features,
                                             stats_years=train_years)
        self.test_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=test_years,
                                              n_leading_observations=self.n_leading_observations,
                                              n_leading_observations_test_adjustment=self.n_leading_observations_test_adjustment,
                                              crop_side_length=self.crop_side_length,
                                              load_from_hdf5=self.load_from_hdf5, is_train=False,
                                              remove_duplicate_features=self.remove_duplicate_features,
                                              features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                              return_metadata=self.return_metadata,
                                              eval_pad_multiple=self.eval_pad_multiple,
                                              standardize_features=self.standardize_features,
                                              stats_years=train_years)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        batch_size = self.eval_batch_size or self.batch_size
        return self._loader(self.val_dataset, shuffle=False, batch_size=batch_size)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False, batch_size=1)

    def predict_dataloader(self):
        batch_size = self.eval_batch_size or self.batch_size
        return self._loader(self.val_dataset, shuffle=False, batch_size=batch_size)

    @staticmethod
    def split_fires(data_fold_id):
        """_summary_ Split the years into train/val/test set.

        Args:
            data_fold_id (_type_): _description_ Index of the respective split to choose, see method body for details.

        Returns:
            _type_: _description_
        """

        folds = [(2018, 2019, 2020, 2021),
                 (2018, 2019, 2021, 2020),
                 (2018, 2020, 2019, 2021),
                 (2018, 2020, 2021, 2019),
                 (2018, 2021, 2019, 2020),
                 (2018, 2021, 2020, 2019),
                 (2019, 2020, 2018, 2021),
                 (2019, 2020, 2021, 2018),
                 (2019, 2021, 2018, 2020),
                 (2019, 2021, 2020, 2018),
                 (2020, 2021, 2018, 2019),
                 (2020, 2021, 2019, 2018)]

        train_years = list(folds[data_fold_id][:2])
        val_years = list(folds[data_fold_id][2:3])
        test_years = list(folds[data_fold_id][3:4])

        print(
            f"Using the following dataset split:\nTrain years: {train_years}, Val years: {val_years}, Test years: {test_years}")

        return train_years, val_years, test_years
