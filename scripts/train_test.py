import gc

from fire import Fire
import torch
from omegaconf import OmegaConf
import sys
sys.path.append("..")
from ldcast.models.autoenc import autoenc, encoder
from ldcast.models.genforecast import analysis, training, unet
from ldcast.features import batch, patches, split, transform
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from train_nowcaster import setup_data
import netCDF4 as nc
import pickle
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from fire import Fire
import xarray as xr
import random


def setup_model(
    num_timesteps=5,
    model_dir="../results/",
    autoenc_weights_fn="/home/dupf/ldcast/ldcast-master/models/autoenc_train/autoenc-11-26_2037-0.02.pt",
    use_obs=True,
    use_nwp=False,
    nwp_input_patches=4,
    num_nwp_vars=9,
    lr=1e-4
):
    enc = encoder.SimpleConvEncoder()
    dec = encoder.SimpleConvDecoder()
    autoencoder_obs = autoenc.AutoencoderKL(enc, dec)
    autoencoder_obs.load_state_dict(torch.load(autoenc_weights_fn))

    autoencoders = []
    input_patches = []
    input_size_ratios = []
    embed_dim = []
    analysis_depth = []
    if use_obs:
        autoencoders.append(autoencoder_obs)
        input_patches.append(1)
        input_size_ratios.append(1)
        embed_dim.append(128)
        analysis_depth.append(4)
    if use_nwp:
        autoencoder_nwp = autoenc.DummyAutoencoder(width=num_nwp_vars)
        autoencoders.append(autoencoder_nwp)
        input_patches.append(nwp_input_patches)
        input_size_ratios.append(2)
        embed_dim.append(32)
        analysis_depth.append(2)

    analysis_net = analysis.AFNONowcastNetCascade(
        autoencoders,
        input_patches=input_patches,
        input_size_ratios=input_size_ratios,
        train_autoenc=False,
        output_patches=num_timesteps,
        cascade_depth=3,
        embed_dim=embed_dim,
        analysis_depth=analysis_depth
    )

    model = unet.UNetModel(in_channels=autoencoder_obs.hidden_width,
        model_channels=256, out_channels=autoencoder_obs.hidden_width,
        num_res_blocks=2, attention_resolutions=(1,2),
        dims=3, channel_mult=(1, 2, 4), num_heads=8,
        num_timesteps=num_timesteps, context_ch=analysis_net.cascade_dims
    )

    (ldm, trainer) = training.setup_genforecast_training(
        model, autoencoder_obs, context_encoder=analysis_net,
        model_dir=model_dir, lr=lr
    )
    gc.collect()
    return (ldm, trainer)



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
            if dataset.variables[list(dataset.variables.keys())[1]].shape[0] > 24:
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

        # while 'observation' not in batch.keys() and (idx + self.count) < len(self.data):
        #     key = keys[idx + self.count]
        #     self.count = self.count + 1
        #     batch = self.load_nc_data(self.data[key])

        # 观测目标
        observation_dict = batch['observation']
        target_shape = (256, 256)
        obs_data = self.reflect_padding(observation_dict[0].filled(0), target_shape)
        obs_data = observation_transform(obs_data)
        observation_tuple = (obs_data.reshape(1, 4, 256, 256),
                             observation_dict[1])
        pred_batch = [observation_tuple]

        # 预测目标
        prediction_matrix = batch['prediction'][0]
        prediction_matrix_np = prediction_matrix.filled(0)
        tar_data = self.reflect_padding(prediction_matrix_np, target_shape)
        tar_data = prediction_transform(tar_data)
        target_batch = tar_data.reshape(1, 20, 256, 256)

        # print(np.isnan(pred_batch[0][0]).any(),np.isnan(target_batch).any())
        return pred_batch, target_batch


    def load_nc_data(self,data):
        patches = data[0][1]
        patch_times = data[1][1]

        max_start_index = len(patch_times) - 24

        observation_data_list = []
        prediction_data_list = []
        observation_times_list = []
        prediction_times_list = []

        # 确保数据长度足够
        count = 0
        while count < 1:
            random_index = random.randint(0, max_start_index)  # 确保后面有24个时刻的数据

            # 获取前四个时刻的观测数据
            observation_data = patches[random_index:random_index + 4]
            # 获取接下来20个时刻的预测数据
            prediction_data = patches[random_index + 4:random_index + 24]

            # 检查观测数据和预测数据的有效性
            # if self.check_submatrices(observation_data) or self.check_submatrices(prediction_data):
            #     continue
            if (np.max(observation_data) - np.min(observation_data) == 0 or
                    np.max(prediction_data) - np.min(prediction_data) == 0):
                continue

            # 获取对应的时间序列
            # observation_times = patch_times[random_index:random_index + 4]
            observation_times = np.array([-3.0, -2.0, -1.0, 0.0])
            prediction_times = patch_times[random_index + 4:random_index + 24]

            # 将数据添加到列表中
            observation_data_list.append(observation_data)
            prediction_data_list.append(prediction_data)
            observation_times_list.append(observation_times)
            prediction_times_list.append(prediction_times)

            count += 1

        # 转换列表为所需形状的numpy数组
        # observation_data_array = np.stack(observation_data_list, axis=0)  # 4*256*256
        observation_data_array = observation_data_list[0]
        # prediction_data_array = np.stack(prediction_data_list, axis=0)  # 4*5*256*256
        prediction_data_array = prediction_data_list[0]
        # prediction_data_array = prediction_data_array.reshape(-1, *prediction_data_array.shape[2:])  # 20*256*256

        # 数据归一化并乘以250，取整
        observation_data_array = ((observation_data_array - observation_data_array.min()) / (
                observation_data_array.max() - observation_data_array.min()) * 250).astype(np.uint8)
        prediction_data_array = ((prediction_data_array - prediction_data_array.min()) / (
                prediction_data_array.max() - prediction_data_array.min()) * 250).astype(np.uint8)


        # 将时间序列也存储为numpy数组（不强制要求与数据匹配形状，只要形式相同）
        # observation_times_array = np.array(observation_times_list)  # 4
        observation_times_array = observation_times_list[0]
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
            "train": "../cache/train.pkl",
            "valid": "../cache/valid.pkl",
            "test": "../cache/test.pkl",
        }
    moudle = TyphoonDataModule(
        batch_size=batch_size,
        train_sampler_file=sampler_file['train'],
        valid_sampler_file=sampler_file['valid'],
        test_sampler_file=sampler_file['test']
    )
    moudle.setup()
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

# ------------------------------------------------------------------------------------------------------
def train(
    future_timesteps=20,
    use_obs=True,
    use_nwp=False,
    sample_shape=(4,4),
    batch_size=8,
    sampler=None,
    ckpt_path=None,
    initial_weights=None,
    strict_weights=True,
    model_dir=None,
    lr=1e-4
):

    if sampler is None:
        sampler_file = None
    else:
        sampler_file = {
            s: f"{sampler}_{s}.pkl" for s in ["test", "train", "valid"]
        }

    print("Loading data...")
    datamodule = setup_TCIR_data(sampler_file,batch_size)
    # datamodule.setup()

    print("Setting up model...")
    (model, trainer) = setup_model(
        num_timesteps=future_timesteps//4,
        use_obs=use_obs,
        use_nwp=use_nwp,
        model_dir=model_dir,
        lr=lr
    )
    if initial_weights is not None:
        print(f"Loading weights from {initial_weights}...")
        model.load_state_dict(
            torch.load(initial_weights, map_location=model.device),
            strict=strict_weights
        )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

def main(config=None, **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)
    train(**config)

if __name__ == "__main__":
    Fire(main)
