# ----------------------------------------------------让数据>0-----------------------------
import os
import numpy as np
from netCDF4 import Dataset

# 源文件夹路径
source_folder_path = '/home/data/dupf/dpf_data/test_1109_1054'
# 目标文件夹路径
target_folder_path = '/home/data/dupf/dpf_data/target_dic'

# 确保目标文件夹存在
os.makedirs(target_folder_path, exist_ok=True)

# 遍历源文件夹中的所有 .nc 文件
for filename in os.listdir(source_folder_path):
    if filename.endswith('.nc'):
        source_file_path = os.path.join(source_folder_path, filename)
        target_file_path = os.path.join(target_folder_path, filename)

        # 打开源 .nc 文件
        with Dataset(source_file_path, 'r') as src_nc_file:
            # 创建目标 .nc 文件
            with Dataset(target_file_path, 'w', format='NETCDF4') as dst_nc_file:
                # 复制全局属性
                dst_nc_file.setncatts(src_nc_file.__dict__)

                # 复制维度
                for name, dimension in src_nc_file.dimensions.items():
                    dst_nc_file.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

                # 复制变量
                for name, variable in src_nc_file.variables.items():
                    dst_var = dst_nc_file.createVariable(name, variable.datatype, variable.dimensions)
                    dst_var.setncatts(variable.__dict__)

                    # 复制数据，如果变量是 patches 则进行处理
                    if name == 'patches':
                        data = variable[:]
                        data[data < 0] = 0  # 将小于0的值变为0
                        dst_var[:] = data
                    else:
                        dst_var[:] = variable[:]

        print(f"Processed {filename} and saved to {target_file_path}")

# ----------------------------------------------------数据归一化-----------------------------
# import os
# import numpy as np
# from netCDF4 import Dataset
#
# # 目标文件夹路径
# target_folder_path = '/home/data/dupf/dpf_data/target_dic'
#
# # 确保目标文件夹存在
# os.makedirs(target_folder_path, exist_ok=True)
#
# # 遍历目标文件夹中的所有 .nc 文件
# for filename in os.listdir(target_folder_path):
#     if filename.endswith('.nc'):
#         file_path = os.path.join(target_folder_path, filename)
#
#         # 打开 .nc 文件
#         with Dataset(file_path, 'r+') as nc_file:
#             # 读取 patches 矩阵
#             patches = nc_file.variables['patches'][:]
#
#             # 将 patches 矩阵中小于0的值变为0
#             patches[patches < 0] = 0
#
#             # 对 patches 矩阵进行归一化处理
#             min_val = patches.min()
#             max_val = patches.max()
#             normalized_patches = (patches - min_val) / (max_val - min_val)
#
#             # 写回归一化后的 patches 矩阵
#             nc_file.variables['patches'][:] = normalized_patches
#
#         print(f"Processed and normalized {filename}")
