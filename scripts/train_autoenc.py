import gc
import gzip
import os
import pickle
import sys
from fire import Fire
import numpy as np
from omegaconf import OmegaConf
sys.path.append("..")
from ldcast.features import batch, patches, split, transform
from ldcast.models.autoenc import encoder, training
# from ldcast.features.transform import default_rainrate_transform
import netCDF4 as nc
import pickle
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from fire import Fire
import xarray as xr
import random


file_dir = os.path.dirname(os.path.abspath(__file__))


def setup_model(
    model_dir="../models/autoenc_train"
):
    enc = encoder.SimpleConvEncoder()
    dec = encoder.SimpleConvDecoder()
    (autoencoder, trainer) = training.setup_autoenc_training(
        encoder=enc,
        decoder=dec,
        model_dir=model_dir
    )
    gc.collect()
    return (autoencoder, trainer)


# ---------------------------------代码修改---------------------------------
from torch.utils.data import IterableDataset
import numpy as np

class CustomIterableDataset(Dataset):
    def __init__(self,
        data_files,
        iterations,
        used_variables,
        interval_secs,
        variables,
        batch_size
        ):
        with open(data_files, 'rb') as f:
            self.data_files = pickle.load(f)
        self.iterations = iterations
        self.used_variables = used_variables
        self.interval_secs = interval_secs
        self.variables = variables
        self.data = self._load_all_data()
        self.count = 0

    def _load_all_data(self):
        data = {}
        for filepath in self.data_files:
            dataset = nc.Dataset(filepath)
            typhoon_id = filepath.split('/')[-1].split('_')[1].split('.')[0]
            typhoon_data = []
            if dataset.variables[list(dataset.variables.keys())[1]].shape[0] > 8:
                for var_name in dataset.variables:
                    var_data = dataset.variables[var_name][:]
                    typhoon_data.append((var_name, var_data))
                data[typhoon_id] = typhoon_data
                dataset.close()
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # 读取scale数组
        nc_file = nc.Dataset('/home/data/dupf/dpf_data/RZC/patches_RZC_201804.nc', 'r')
        scale = np.array(nc_file.variables['scale'][:])
        nc_file.close()

        # 应用变换
        observation_transform = transform.default_rainrate_transform(scale)
        prediction_transform = transform.default_rainrate_transform(scale)

        keys = list(self.data.keys())
        key = keys[idx]
        batch = self.load_nc_data(self.data[key])

        # 观测目标
        observation_dict = batch['observation']
        target_shape = (256, 256)
        obs_data = self.reflect_padding(observation_dict[0].filled(0), target_shape)
        obs_data = observation_transform(obs_data)
        observation_tuple = (obs_data.reshape(1, 8, 256, 256),
                             observation_dict[1])
        pred_batch = [observation_tuple]

        # 预测目标
        prediction_matrix = batch['prediction'][0]
        prediction_matrix_np = prediction_matrix.filled(0)
        tar_data = self.reflect_padding(prediction_matrix_np, target_shape)
        tar_data = prediction_transform(tar_data)
        target_batch = tar_data.reshape(1, 8, 256, 256)


        # pred_batch[0][0] = observation_data_array
        # target_batch = prediction_data_array
        # pred_batch = (pred_batch - pred_batch.min()) / (pred_batch.max() - pred_batch.min())
        # target_batch = (target_batch - target_batch.min()) / (target_batch.max() - target_batch.min())

        return pred_batch, target_batch

    def load_nc_data(self, data):
        patches = data[0][1]
        patch_times = data[1][1]

        max_start_index = len(patch_times) - 8

        observation_data_list = []
        prediction_data_list = []
        observation_times_list = []
        prediction_times_list = []

        count = 0
        while count < 1:
            random_index = random.randint(0, max_start_index)

            # 获取观测数据和预测数据
            observation_data = patches[random_index: random_index + 8]
            prediction_data = patches[random_index: random_index + 8]

            if np.max(observation_data) - np.min(observation_data) == 0 or np.max(prediction_data) - np.min(
                    prediction_data) == 0:
                continue

            # 获取对应的时间序列
            observation_times = patch_times[random_index: random_index + 8]
            prediction_times = patch_times[random_index: random_index + 8]

            # 将数据添加到列表中
            observation_data_list.append(observation_data)
            prediction_data_list.append(prediction_data)
            observation_times_list.append(observation_times)
            prediction_times_list.append(prediction_times)

            count += 1

        # 转换列表为所需形状的numpy数组
        # observation_data_array = np.stack(observation_data_list, axis=0)  # 4*256*256
        observation_data_array = observation_data_list[0]
        prediction_data_array = prediction_data_list[0]
        # prediction_data_array = np.stack(prediction_data_list, axis=0)  # 4*5*256*256

        # 数据归一化并乘以250，取整
        observation_data_array = ((observation_data_array - observation_data_array.min()) / (
                    observation_data_array.max() - observation_data_array.min()) * 250).astype(np.uint8)
        prediction_data_array = ((prediction_data_array - prediction_data_array.min()) / (
                    prediction_data_array.max() - prediction_data_array.min()) * 250).astype(np.uint8)


        # 将时间序列也存储为numpy数组（不强制要求与数据匹配形状，只要形式相同）
        observation_times_array = np.array(observation_times_list)  # 4
        prediction_times_array = np.array(prediction_times_list)

        # 用字典保存观测数据和预测数据
        data_split = {
            'observation': {
                0: observation_data_array,
                1: observation_times_array
            },
            'prediction': {
                0: prediction_data_array,
                1: prediction_times_array
            }
        }

        return data_split

    def reflect_padding(self,matrix, target_shape):
        original_shape = matrix.shape[-2:]
        pad_height = target_shape[0] - original_shape[0]
        pad_width = target_shape[1] - original_shape[1]

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        if matrix.ndim == 2:
            padded_matrix = np.pad(matrix, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                                   constant_values=0)
        else:
            padded_matrix = np.pad(matrix, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                                   constant_values=0)
        return padded_matrix


class TyphoonDataModule(pl.LightningDataModule):
    def __init__(self, batch_size,
        train_sampler_file, valid_sampler_file, test_sampler_file,
        train_iterations=500, val_iterations=200, test_iterations=500
    ):
        super().__init__()
        self.batch_size = batch_size
        self.interval = 10800
        self.train_files = train_sampler_file
        self.val_files = valid_sampler_file
        self.test_files = test_sampler_file
        self.train_iterations = train_iterations
        self.val_iterations = val_iterations
        self.test_iterations = test_iterations
        self.used_variables = {'observation','prediction'}
        self.interval_secs = 10800

        var_data = {
            'source': ['TCIR'],
            'timesteps': np.arange(-10, 1)
        }

        # 定义 T 键的值（此处以示例数据进行展示）
        T_var_data = {
            'source': ['TCIR'],
            'timesteps': np.arange(1, 11)  # 例如从 1 到 11
        }
        # 创建包含两个键值 'O' 和 'T' 的字典
        variables = {
            'O': var_data,
            'T': T_var_data
        }
        self.variables = variables


    def setup(self, stage=None):
        self.train_dataset = CustomIterableDataset(self.train_files,self.train_iterations,self.used_variables,
                                                   self.interval_secs,self.variables,self.batch_size)
        self.val_dataset = CustomIterableDataset(self.val_files,self.train_iterations,self.used_variables,
                                                 self.interval_secs,self.variables,self.batch_size)
        self.test_dataset = CustomIterableDataset(self.test_files,self.train_iterations,self.used_variables,
                                                  self.interval_secs,self.variables,self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

def setup_TCIR_data(sampler_file, batch_size):
    if sampler_file is None:
        sampler_file = {
            "train": "../cache/autoenc_train.pkl",
            "valid": "../cache/autoenc_valid.pkl",
            "test": "../cache/autoenc_test.pkl",
        }
    moudle = TyphoonDataModule(
        batch_size=batch_size,
        train_sampler_file=sampler_file['train'],
        valid_sampler_file=sampler_file['valid'],
        test_sampler_file=sampler_file['test']
    )
    return moudle


def load_data_filenames(pkl_file):
    with open(pkl_file, 'rb') as f:
        data_files = pickle.load(f)
    return data_files

def load_and_preprocess(file_path):
    ds = xr.open_dataset(file_path)
    # 假设降水数据的变量名为"precipitation"
    precipitation_data = ds['precipitation'].values
    # 这里可以添加更多的预处理步骤
    return precipitation_data




def train(
    var="RZC",
    batch_size=8,
    sampler_file=None,
    num_timesteps=8,
    chunks_file="/home/data/dupf/dpf_data/split_chunks.pkl.gz",
    model_dir=None,
    ckpt_path=None
):
    print("Loading data...")
    # datamodule = setup_data(
    #     var=var, batch_size=batch_size, sampler_file=sampler_file,
    #     num_timesteps=num_timesteps, chunks_file=chunks_file
    # )
    datamodule = setup_TCIR_data(
        batch_size=batch_size, sampler_file=sampler_file
    )
    # datamodule.setup()

    print("Setting up model...")
    (model, trainer) = setup_model(model_dir=model_dir)

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def main(config=None, **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)
    train(**config)


if __name__ == "__main__":
    Fire(main)
