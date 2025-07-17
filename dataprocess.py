import os
import netCDF4
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import h5py
from datetime import datetime, timedelta
import dask


# data_path = "/home/data/dupf/dpf_data/TCIR-ATLN_EPAC_WPAC.h5"
data_path = "/home/data/dupf/dpf_data/TCIR-CPAC_IO_SH.h5"

data_info = pd.read_hdf(data_path, key="info", mode='r')
# print(data_info.head())


with h5py.File(data_path, 'r') as hf:
    data_matrix = hf['matrix'][:]

channel_four_data = data_matrix[:, :, :, 0]
# 划分出每块区域
regions = data_info['data_set']  # 'data_set' 是包含区域信息的列
# # 根据区域信息分组数据
# region_data = {}
unique_regions = pd.unique(regions)

typhoon_ids = data_info['ID']  # 'ID' 是包含台风 ID 的列
# 根据台风 ID 进一步划分数据
region_typhoon_data = {}
for region in unique_regions:
    region_typhoon_data[region] = {}
    region_indices = regions.index[regions == region].tolist()
    region_info = data_info.loc[region_indices]
    for typhoon_id in region_info['ID'].unique():
        typhoon_indices = region_info.index[region_info['ID'] == typhoon_id].tolist()
        region_typhoon_data[region][typhoon_id] = channel_four_data[typhoon_indices]
# for region, typhoons in region_typhoon_data.items():
#     print(f'Region: {region}')
#     for typhoon_id, data in typhoons.items():
#         print(f'  Typhoon ID: {typhoon_id}, Data Shape: {data.shape}')

first_region = list(region_typhoon_data.keys())[0]
first_typhoon_id = list(region_typhoon_data[first_region].keys())[0]
first_typhoon_data = region_typhoon_data[first_region][first_typhoon_id]
new_dict = {first_region: {first_typhoon_id: first_typhoon_data}}

patch_data = new_dict[first_region][first_typhoon_id]
patch_data = np.stack(patch_data, axis=0)

# 假设 data_info 已经存在并且包含 'time' 列
data_info['time'] = pd.to_datetime(data_info['time'], format='%Y%m%d%H')

# 取前77个转换后的 datetime 值
data_info_77 = data_info.iloc[:77]
time_77 = pd.to_datetime(data_info['time'].iloc[:77], format='%Y%m%d%H')


# 定义输出文件路劲名
def save_var(new_dict, patch_data, patch_times):
    # 获取第一个 region 的键和值
    first_region_key = next(iter(new_dict))
    first_region_value = new_dict[first_region_key]

    # 获取第一个 typhoon 的键和值
    first_typhoon_key = next(iter(first_region_value))

    out_fn = f"{first_region_key}_{first_typhoon_key}.nc"
    out_fn = os.path.join(out_dir, out_fn)
    # try:
    #     time = epoch + timedelta(seconds=int(patch_times[0]))
    #     var_scale = reader.get_scale(time)
    # except (AttributeError, KeyError):
    #     var_scale = None if (scale is None) else scale
    #     pass
    save_patches(patch_data, patch_times, out_fn)

# 使用函数保存nc文件，可运行
def save_patches(patch_data, patch_times, out_fn, zero_value=0):
    with Dataset(out_fn, 'w') as ds:
        # 创建维度
        dim_patch = ds.createDimension("dim_patch", patch_data.shape[0])
        dim_height = ds.createDimension("dim_height", patch_data.shape[1])
        dim_width = ds.createDimension("dim_width", patch_data.shape[2])

        # 设置变量属性
        var_args = {"zlib": True, "complevel": 1}
        #
        # 设置分块大小（可选，但有助于大文件的性能）
        chunksizes = (min(2 ** 10, patch_data.shape[0]), patch_data.shape[1], patch_data.shape[2])

        # 创建变量
        var_patch = ds.createVariable("patches", patch_data.dtype,
                                      ("dim_patch", "dim_height", "dim_width"),
                                      chunksizes=chunksizes, **var_args)

        # 写入数据
        var_patch[:] = patch_data

        var_patch_time = ds.createVariable("patch_times", patch_times.dtype,
                                           ("dim_patch",), **var_args)
        var_patch_time[:] = patch_times


        ds.zero_value = zero_value

# save_var(new_dict)

# 获取patch_data:一个台风对应的所有数据,patch_time
def get_patches(i,j,
        epoch=datetime(1970, 1, 1), postproc=None,
        pool=None, min_nonzeros_to_include=1
):
    global time_last

    region = list(region_typhoon_data.keys())[i]
    typhoon_id = list(region_typhoon_data[region].keys())[j]
    typhoon_data = region_typhoon_data[region][typhoon_id]
    new_dict = {region: {typhoon_id: typhoon_data}}
    patch_data = new_dict[region][typhoon_id]
    patch_data = np.stack(patch_data, axis=0)
    patches = []
    patch_times = []


    time_77 = pd.to_datetime(data_info['time'].iloc[time_last:time_last + patch_data.shape[0]], format='%Y%m%d%H')
    time_secs = np.array([(t - epoch).total_seconds() for t in time_77])


    for batch_idx in range(patch_data.shape[0]):
        patch = patch_data[batch_idx]  # 直接使用201x201的原始数据
        patches.append(patch)
        patch_times.append(time_secs[batch_idx])

    patches = np.array(patches)
    patch_times = np.array(patch_times)

    # new_shape_first_dim = patches.shape[0] * patches.shape[1]

    # 使用 reshape 函数改变数组的形状
    # reshaped_patches = patches.reshape(new_shape_first_dim, 32, 32)

    variables = "RZC"

    # 假设 data_info 已经存在并且包含 'time' 列
    data_info['time'] = pd.to_datetime(data_info['time'], format='%Y%m%d%H')


    # 取前77个转换后的 datetime 值

    time_last = patch_data.shape[0] + time_last
    save_var(new_dict, patches, patch_times)


import json
def save_typhoon_tracks(data_info, typhoon_id='200619W', start_idx=9, end_idx=17, output_path=None):
    """
    保存指定台风的轨迹数据为JSON文件
    
    Args:
        data_info (pd.DataFrame): 包含台风信息的DataFrame（需包含'ID', 'time', 'lon', 'lat'列）
        typhoon_id (str): 目标台风ID，默认'200619W'
        start_idx (int): 记录起始索引（从0开始），默认9（第10条记录）
        end_idx (int): 记录结束索引（不包含），默认17（第17条记录）
        output_path (str): 输出JSON文件路径，默认使用台风ID生成路径
    """
    # 校验台风是否存在
    if typhoon_id not in data_info['ID'].values:
        raise ValueError(f"台风ID '{typhoon_id}' 不存在于data_info中")
    
    # 筛选指定台风数据
    typhoon_data = data_info[data_info['ID'] == typhoon_id]
    
    # 校验索引范围
    if end_idx > len(typhoon_data):
        raise IndexError(f"end_idx {end_idx} 超过台风'{typhoon_id}'的记录总数（{len(typhoon_data)}）")
    
    # 提取指定索引的子集（[start_idx, end_idx)）
    typhoon_subset = typhoon_data.iloc[start_idx:end_idx][['time', 'lon', 'lat']]
    
    # 转换为字典列表格式，并将时间字段从Timestamp转为指定格式字符串
    track_data = typhoon_subset.to_dict('records')
    # 关键修改：将time字段的Timestamp对象转为YYYYMMDDHH格式字符串
    for record in track_data:
        record['time'] = record['time'].strftime("%Y%m%d%H")  # 格式示例：2006100415（年+月+日+时）
    
    # 生成默认输出路径（若未指定）
    if output_path is None:
        output_path = f"typhoon_{typhoon_id}_tracks.json"

    # 保存为JSON文件
    with open(output_path, 'w') as f:
        json.dump(track_data, f, indent=4)  # 现在time字段是字符串，可正常序列化
    
    print(f"台风 {typhoon_id} 轨迹数据已保存至：{output_path}")

# 获取其他台风（如'200614W'）的第5-10条记录
save_typhoon_tracks(
    data_info=data_info,
    typhoon_id='201603S',
    start_idx=0,
    end_idx=8,
    # output_path='/home/dupf/code/ldcast-master/typhoon_200614W_tracks.json'
)

# # 打印详细信息
# print("台风200619W第10-17条记录详细信息:")
# print(typhoon_subset)
# print(f"\n共获取 {len(typhoon_subset)} 条记录")

# # 如果需要保存到文件
# # typhoon_200614W.to_csv('/home/dupf/code/ldcast-master/typhoon_200614W_info.csv', index=False)

