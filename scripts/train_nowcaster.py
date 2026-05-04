from datetime import timedelta
import gc
import gzip
import os
import pickle
import netCDF4 as nc
import numpy as np
import numpy.ma as ma

from ldcast.features import batch, patches, split, transform

file_dir = os.path.dirname(os.path.abspath(__file__))


def setup_data(
        use_obs=True,
        use_nwp=False,
        obs_vars=("RZC",),
        nwp_vars=(
                "cape", "cin", "rate-cp", "rate-tp", "t2m",
                "tclw", "tcwv", "u", "v"
        ),
        nwp_lags=(0, 12),
        target_var="RZC",
        batch_size=8,
        past_timesteps=4,
        future_timesteps=20,
        # timestep_secs=10800,
        timestep_secs=300,
        nwp_timestep_secs=3600,
        sampler_file=None,
        chunks_file="../data/split_chunks.pkl.gz",
        sample_shape=(4, 4)
):
    target = target_var + "-T"
    predictors_obs = [v + "-O" for v in obs_vars]
    predictors = []
    if use_obs:
        predictors += predictors_obs
    if use_nwp:
        predictors.append("nwp")

    variables = {
        target: {
            "sources": [target_var],
            "timesteps": np.arange(1, future_timesteps + 1),
        }
    }
    for (var, raw_var) in zip(predictors_obs, obs_vars):
        variables[var] = {
            "sources": [raw_var],
            "timesteps": np.arange(-past_timesteps + 1, 1)
        }
    nwp_t1 = int(np.ceil(future_timesteps * timestep_secs / nwp_timestep_secs)) + 2
    nwp_range = np.arange(nwp_t1)
    variables["nwp"] = {
        "sources": nwp_vars,
        "timesteps": nwp_range,
        "timestep_secs": nwp_timestep_secs
    }

    # determine which raw variables are needed, then load them
    raw_vars = set.union(
        *(set(variables[v]["sources"]) for v in predictors_obs + [target])
    )
    if use_nwp:
        for raw_var_base in variables["nwp"]["sources"]:
            raw_vars.update(f"{raw_var_base}-{lag}" for lag in nwp_lags)

    # raw = {
    #     var: patches.load_all_patches(
    #         os.path.join(file_dir, f"/home/dupf/ldcast/ldcast-master/data/{var}/"), var
    #     )
    #     for var in raw_vars
    # }
    # # Load pregenerated train/valid/test split data.
    # # These can be generated with features.split.get_chunks()
    # with gzip.open(os.path.join(file_dir, chunks_file), 'rb') as f:
    #     chunks = pickle.load(f)
    # (raw, _) = split.train_valid_test_split(raw, var, chunks=chunks)

    # '''
    def process_nc_files(folder_path):
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.nc')])

        train_files = []
        test_files = []
        val_files = []

        for file_name in files:
            year = int(file_name.split('_')[3][:4])
            file_path = os.path.join(folder_path, file_name)

            if year <= 2003:
                train_files.append(file_path)
            elif year == 2004:
                test_files.append(file_path)
            elif year == 2005:
                val_files.append(file_path)

        return train_files, test_files, val_files

    def process_nc_files1(folder_path):
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.nc')])

        train_files = []
        test_files = []
        val_files = []

        for file_name in files:
            year = int(file_name.split('_')[2][:4])
            file_path = os.path.join(folder_path, file_name)

            if  year == 2018:
                train_files.append(file_path)
            elif year == 2019:
                test_files.append(file_path)
            elif year == 2020:
                val_files.append(file_path)

        return train_files, test_files, val_files

    # def split_files(file_list, valid_frac=0.1, test_frac=0.2, random_seed=None):
    #     rng = np.random.RandomState(seed=random_seed)
    #     rng.shuffle(file_list)
    #
    #     num_files = len(file_list)
    #     num_valid = int(round(num_files * valid_frac))
    #     num_test = int(round(num_files * test_frac))
    #     num_train = num_files - num_valid - num_test
    #
    #     train_files = file_list[:num_train]
    #     valid_files = file_list[num_train:num_train + num_valid]
    #     test_files = file_list[num_train + num_valid:]
    #
    #     return train_files, valid_files, test_files

    def load_nc_file(file_path):
        with nc.Dataset(file_path, 'r') as data:
            patches = data.variables['patches'][:]
            patch_coords = data.variables['patch_coords'][:]
            patch_times = data.variables['patch_times'][:]
            zero_patch_coords = data.variables['zero_patch_coords'][:]
            zero_patch_times = data.variables['zero_patch_times'][:]
            scale = data.variables['scale'][:]
        return patches, patch_coords, patch_times, zero_patch_coords, zero_patch_times, scale

    def load_split(files):
        patches_list = []
        times_list = []
        patch_coords_list = []
        zero_patch_coords_list = []
        zero_patch_times_list = []
        scale_list = []
        for file in files:
            patches, patch_coords, patch_times, zero_patch_coords, zero_patch_times, scale = load_nc_file(file)
            patches_list.append(patches)
            times_list.append(patch_times)
            patch_coords_list.append(patch_coords)
            zero_patch_coords_list.append(zero_patch_coords)
            zero_patch_times_list.append(zero_patch_times)
            if len(scale_list) == 0:
                scale_list.append(scale)

        return (np.concatenate(patches_list), np.concatenate(patch_coords_list),
                np.concatenate(times_list), np.concatenate(zero_patch_coords_list),
                np.concatenate(zero_patch_times_list), np.concatenate(scale_list))

    def prepare_datasets(data_dir):
        train_files, test_files, val_files = process_nc_files(data_dir)

        train_data = load_split(train_files)
        valid_data = load_split(val_files)
        test_data = load_split(test_files)

        return train_data, valid_data, test_data

    def convert_to_dict(train_data, valid_data, test_data):
        def create_dict(patches, patch_coords, patch_times, zero_patch_coords, zero_patch_times, scale):
            return {
                "patches": patches,
                "patch_coords": patch_coords,
                "patch_times": patch_times,
                "zero_patch_coords": zero_patch_coords,
                "zero_patch_times": zero_patch_times,
                "scale": scale,
            }

        raw = {
            "train": create_dict(*train_data),
            "valid": create_dict(*valid_data),
            "test": create_dict(*test_data),
        }

        return raw

    # 使用示例
    data_dir = '/home/dupf/ldcast/ldcast-master/data/test1'  # Adjust this path to your data directory
    train_data, valid_data, test_data = prepare_datasets(data_dir)
    raw_data_dict = convert_to_dict(train_data, valid_data, test_data)

    def add_rzc_to_dict(data_dict):
        for key in ["train", "valid", "test"]:
            patches = data_dict[key].pop("patches")
            patch_coords = data_dict[key].pop("patch_coords")
            patch_times = data_dict[key].pop("patch_times")
            zero_patch_coords = data_dict[key].pop("zero_patch_coords")
            zero_patch_times = data_dict[key].pop("zero_patch_times")
            scale = data_dict[key].pop("scale")
            data_dict[key]["RZC"] = {
                "patches": patches,
                "patch_coords": patch_coords,
                "patch_times": patch_times,
                "zero_patch_coords": zero_patch_coords,
                "zero_patch_times": zero_patch_times,
                "scale": scale,
            }
        return data_dict

    # 更新字典
    raw = add_rzc_to_dict(raw_data_dict)

    def convert_maskedarrays_to_nparrays(d):
        for key, value in d.items():
            if isinstance(value, dict):
                # 如果值是字典，则递归调用
                convert_maskedarrays_to_nparrays(value)
            elif isinstance(value, ma.masked_array):
                # 如果值是 maskedarray，则转换为 nparray
                d[key] = ma.filled(value, fill_value=np.nan)

    convert_maskedarrays_to_nparrays(raw)


    # for key in ['train', 'valid', 'test']:  # 遍历 train, valid, test 键
    #     if key in raw:
    #         if 'RZC' in raw[key]:
    #             # 获取 'patches' 矩阵
    #             patches_matrix = raw[key]['RZC']['patches']
    #             # 处理矩阵
    #             raw[key]['RZC']['patches'] = process_patches(patches_matrix)
# '''





    transform_rain = lambda: transform.default_rainrate_transform(
        raw["train"]["RZC"]["scale"]
    )
    transform_cape = lambda: transform.normalize_threshold(
        log=True,
        threshold=1.0, fill_value=1.0,
        mean=1.530, std=0.859
    )
    transform_rate_tp = lambda: transform.normalize_threshold(
        log=True,
        threshold=1e-5, fill_value=1e-5,
        mean=-3.831, std=0.650
    )
    transform_wind = lambda: transform.normalize(std=9.44)

    transforms = {
        "RZC-T": transform_rain(),
        "RZC-O": transform_rain(),
        "cape": transform_cape(),
        "cin": transform_cape(),
        "rate-tp": transform_rate_tp(),
        "rate-cp": transform_rate_tp(),
        "t2m": transform.normalize(mean=286.069, std=7.323),
        "tclw": transform.normalize_threshold(
            log=True,
            threshold=0.001, fill_value=0.001,
            mean=-1.486, std=0.638
        ),
        "tcwv": transform.normalize(std=17.307),
        "u": transform_wind(),
        "v": transform_wind()
    }
    transforms["nwp"] = transform.combine([transforms[v] for v in nwp_vars])
    for (var_name, var_data) in variables.items():
        var_data["transform"] = transforms[var_name]

    if sampler_file is None:
        sampler_file = {
            "train": "../cache/sampler_test_train.pkl",
            "valid": "../cache/sampler_test_valid.pkl",
            "test": "../cache/sampler_test_test.pkl",
        }

    bins = np.exp(np.linspace(np.log(0.2), np.log(1), 0))
    # bins = np.exp(np.linspace(np.log(0.2), np.log(50), 10))

    datamodule = split.DataModule(
        variables, raw, predictors, target, target,
        forecast_raw_vars=nwp_vars,
        interval=timedelta(seconds=timestep_secs),
        batch_size=batch_size, sampling_bins=bins,
        time_range_sampling=(-past_timesteps + 1, future_timesteps + 1),
        sampler_file=sampler_file,
        sample_shape=sample_shape,
        valid_seed=1234, test_seed=2345,
    )

    gc.collect()
    return datamodule
