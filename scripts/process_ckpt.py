# -----------------------------查看ckpt文件--------------------------
# import torch
# from pprint import pprint
#
# # 加载检查点文件
# checkpoint_path = '/home/dupf/ldcast/ldcast-master/scripts/lightning_logs/version_2/checkpoints/epoch=32-val_rec_loss=0.0133.ckpt'
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#
# # 打印检查点文件的键
# print("Checkpoint keys:", checkpoint.keys())
#
# # 查看 state_dict 中的内容
# state_dict = checkpoint['state_dict']
# print("\nState_dict keys:")
# pprint(state_dict.keys())
#
# # 检查特定键是否存在于 state_dict 中，并打印它们的形状
# keys_to_check = [
#     "log_var", "encoder.0.proj.weight", "encoder.0.proj.bias", "encoder.0.conv1.weight", "encoder.0.conv1.bias",
#     "encoder.0.conv2.weight", "encoder.0.conv2.bias", "encoder.2.conv1.weight", "encoder.2.conv1.bias",
#     "encoder.2.conv2.weight", "encoder.2.conv2.bias", "encoder.2.norm1.weight", "encoder.2.norm1.bias",
#     "decoder.0.weight", "decoder.0.bias", "decoder.1.conv1.weight", "decoder.1.conv1.bias", "decoder.1.conv2.weight",
#     "decoder.1.conv2.bias", "decoder.1.norm1.weight", "decoder.1.norm1.bias", "decoder.1.norm2.weight",
#     "decoder.1.norm2.bias", "decoder.2.weight", "decoder.2.bias", "decoder.3.proj.weight", "decoder.3.proj.bias",
#     "decoder.3.conv1.weight", "decoder.3.conv1.bias", "decoder.3.conv2.weight", "decoder.3.conv2.bias",
#     "decoder.3.norm1.weight", "decoder.3.norm1.bias", "decoder.3.norm2.weight", "decoder.3.norm2.bias",
#     "to_moments.weight", "to_moments.bias", "to_decoder.weight", "to_decoder.bias"
# ]
#
# for key in keys_to_check:
#     if key in state_dict:
#         print(f"{key} found in state_dict with shape: {state_dict[key].shape}")
#     else:
#         print(f"{key} not found in state_dict")

# ----------------------------保存信息为pt文件-------------------------------------
import torch
from pprint import pprint

# 加载检查点文件
checkpoint_path = '/home/dupf/ldcast/ldcast-master/mytself/lightning_logs/version_1/checkpoints/epoch=62-val_rec_loss=0.0165.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# 打印检查点文件的键
print("Checkpoint keys:", checkpoint.keys())

# 提取 state_dict
state_dict = checkpoint['state_dict']
print("\nState_dict keys:")
pprint(state_dict.keys())

# 保存 state_dict 为 .pt 文件
state_dict_path = '/home/dupf/ldcast/ldcast-master/models/autoenc_train/autoenc-12-02_1751-0.01.pt'
torch.save(state_dict, state_dict_path)

print(f"State_dict has been saved to {state_dict_path}")
