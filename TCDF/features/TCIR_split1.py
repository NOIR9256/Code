import numpy as np
import os
import netCDF4 as nc
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# from . import TCIR_batch

def split_files(file_list, valid_frac=0.1, test_frac=0.2, random_seed=None):
    rng = np.random.RandomState(seed=random_seed)
    rng.shuffle(file_list)

    num_files = len(file_list)
    num_valid = int(round(num_files * valid_frac))
    num_test = int(round(num_files * test_frac))
    num_train = num_files - num_valid - num_test

    train_files = file_list[:num_train]
    valid_files = file_list[num_train:num_train + num_valid]
    test_files = file_list[num_train + num_valid:]

    return train_files, valid_files, test_files

def load_nc_file(file_path):
    with nc.Dataset(file_path, 'r') as data:
        patches = data.variables['patches'][:]
        patch_coords = data.variables['patch_coords'][:]
        patch_times = data.variables['patch_times'][:]
        zero_patch_coords = data.variables['zero_patch_coords'][:]
        zero_patch_times = data.variables['zero_patch_times'][:]
    return patches, patch_coords, patch_times, zero_patch_coords, zero_patch_times

def load_split(files):
    patches_list = []
    times_list = []
    patch_coords_list = []
    zero_patch_coords_list = []
    zero_patch_times_list = []
    for file in files:
        patches, patch_coords, patch_times, zero_patch_coords, zero_patch_times = load_nc_file(file)
        patches_list.append(patches)
        times_list.append(patch_times)
        patch_coords_list.append(patch_coords)
        zero_patch_coords_list.append(zero_patch_coords)
        zero_patch_times_list.append(zero_patch_times)

    return (np.concatenate(patches_list), np.concatenate(times_list),
            np.concatenate(patch_coords_list), np.concatenate(zero_patch_coords_list),
            np.concatenate(zero_patch_times_list))

def prepare_datasets(data_dir, valid_frac=0.1, test_frac=0.2, random_seed=None):
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')]
    train_files, valid_files, test_files = split_files(file_list, valid_frac, test_frac, random_seed)

    train_data = load_split(train_files)
    valid_data = load_split(valid_files)
    test_data = load_split(test_files)

    return train_data, valid_data, test_data

def convert_to_dict(train_data, valid_data, test_data):
    def create_dict(patches, times):
        return {
            "patches": patches,
            "patch_times": times,
        }

    raw = {
        "train": create_dict(*train_data),
        "valid": create_dict(*valid_data),
        "test": create_dict(*test_data),
    }

    return raw

# 使用示例
data_dir = '/home/dupf/ldcast/ldcast-master/data/test'  # Adjust this path to your data directory
train_data, valid_data, test_data = prepare_datasets(data_dir, valid_frac=0.1, test_frac=0.2, random_seed=42)
raw_data_dict = convert_to_dict(train_data, valid_data, test_data)

# 定义一个函数来更新字典
def add_rzc_to_dict(data_dict):
    for key in ["train", "valid", "test"]:
        patches = data_dict[key].pop("patches")
        patch_times = data_dict[key].pop("patch_times")
        data_dict[key]["RZC"] = {
            "patches": patches,
            "patch_times": patch_times
        }
    return data_dict

# 更新字典
updated_data_dict = add_rzc_to_dict(raw_data_dict)

#
# class DataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         variables, raw, predictors, target, primary_var,
#         sampling_bins, sampler_file,
#         batch_size=64,
#         train_epoch_size=1000, valid_epoch_size=200, test_epoch_size=1000,
#         valid_seed=None, test_seed=None,
#         **kwargs
#     ):
#         super().__init__()
#         self.batch_gen = {
#             split: TCIR_batch.BatchGenerator(
#                 variables, raw_var, predictors, target, primary_var,
#                 sampling_bins=sampling_bins, batch_size=batch_size,
#                 sampler_file=sampler_file.get(split),
#                 augment=(split=="train"),
#                 **kwargs
#             )
#             for (split,raw_var) in raw.items()
#         }
#         self.datasets = {}
#         if "train" in self.batch_gen:
#             self.datasets["train"] = TCIR_batch.StreamBatchDataset(
#                 self.batch_gen["train"], train_epoch_size
#             )
#         if "valid" in self.batch_gen:
#             self.datasets["valid"] = TCIR_batch.DeterministicBatchDataset(
#                 self.batch_gen["valid"], valid_epoch_size, random_seed=valid_seed
#             )
#         if "test" in self.batch_gen:
#              self.datasets["test"] = TCIR_batch.DeterministicBatchDataset(
#                 self.batch_gen["test"], test_epoch_size, random_seed=test_seed
#             )
#
#     def dataloader(self, split):
#         return DataLoader(
#             self.datasets[split], batch_size=None,
#             pin_memory=True, num_workers=0
#         )
#
#     def train_dataloader(self):
#         return self.dataloader("train")
#
#     def val_dataloader(self):
#         return self.dataloader("valid")
#
#     def test_dataloader(self):
#         return self.dataloader("test")
#
#
# class TyphoonDataset(Dataset):
#     def __init__(self, data):
#         self.patches, self.times = data
#
#     def __len__(self):
#         return len(self.patches)
#
#     def __getitem__(self, idx):
#         return self.patches[idx], self.times[idx]
#
#
# class TyphoonDataModule(pl.LightningDataModule):
#     def __init__(self, data_dir, batch_size=64, valid_frac=0.1, test_frac=0.2, random_seed=None):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.valid_frac = valid_frac
#         self.test_frac = test_frac
#         self.random_seed = random_seed
#
#     def setup(self, stage=None):
#         train_data, valid_data, test_data = prepare_datasets(self.data_dir, self.valid_frac, self.test_frac,
#                                                              self.random_seed)
#         self.train_dataset = TyphoonDataset(train_data)
#         self.valid_dataset = TyphoonDataset(valid_data)
#         self.test_dataset = TyphoonDataset(test_data)
#
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
#
#     def val_dataloader(self):
#         return DataLoader(self.valid_dataset, batch_size=self.batch_size)
#
#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size)

# data_module = TyphoonDataModule(data_dir='/home/dupf/ldcast/ldcast-master/data/TCIR_data', batch_size=64)
# data_module.setup()
#
# train_loader = data_module.train_dataloader()
# valid_loader = data_module.val_dataloader()
# test_loader = data_module.test_dataloader()