from modules import UNet_conditional
from train_sample import Diffusion
import torch
import numpy
import einops
import matplotlib.pyplot as plt
from train_TCIR import CustomIterableDataset
from torch.utils.data import DataLoader
import numpy as np
device = "cuda:0"


net = UNet_conditional(c_in=4, c_out=4, device=device)
net.to(device)
diffusion = Diffusion(img_size=256, device=device)
net.load_state_dict(torch.load("/home/dupf/code/ldcast-master/conditional-ddpm/save_ckpt/checkcpoint_or.pth"))
# checkpoint = torch.load("/home/dupf/code/ldcast-master/conditional-ddpm/save_ckpt/checkcpoint_or.pth")
# net.load_state_dict(checkpoint["model_state_dict"])
# 读取数据
# radar = numpy.load("/home/dupf/code/ldcast-master/conditional-ddpm/datasets/ex2.npy")

sampler_file = None
if sampler_file is None:
    sampler_file = {
        "train": "/home/dupf/code/ldcast-master/cache/train.pkl",
        "valid": "/home/dupf/code/ldcast-master/cache/valid.pkl",
        "test": "/home/dupf/code/ldcast-master/cache/test.pkl",
    }
test_dataset = CustomIterableDataset(sampler_file['test'], 1)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=True, pin_memory=True
)

for features0, labels in test_loader:
    tagert = features0.to(device)  # Get condition
    lables = labels.to(device)

min_tagert, max_tagert = tagert.min().to(device), tagert.max().to(device)
condition = 2 * (tagert - min_tagert) / (max_tagert - min_tagert) - 1

# '''
# 归一化，tagert条件：前四个时刻；lable：预测目标：后四个时刻groundtruth
# tagert = (radar[0:4, :, :, :] - 0.4202) / 0.8913
# lable = (radar[4:8, :, :, :] - 0.4202) / 0.8913
# t c w h ->  c t w h
# tagert = einops.rearrange(tagert, " t c w h ->  c t w h")
# lable = einops.rearrange(lable, " t c w h ->  c t w h")
# numpy转torch
# tagert = torch.from_numpy(tagert)
# 将tagert条件复制成两遍，lable复制两遍后以通道合并
tagert = einops.repeat(condition,"c t w h-> a c t w h",a =2)
lable = einops.repeat(lables,"c t w h-> (a c) t w h",a =2)
print(tagert.shape)# b c t c h
# 还是变成(a c) t w h
tagert = torch.squeeze(tagert).to(device)
# print(x.shape)
# 传入模型时传入条件target：a c t w h，生成数是target的第一维
sampled_images = diffusion.sample_condition_decoder_(net, n=tagert.shape [0],image=tagert)
sampled_images = (sampled_images - np.min(sampled_images)) / (np.max(sampled_images) - np.min(sampled_images))
# 还原数据
# sampled_images = sampled_images*0.8913+0.4202
# lable = lable*0.8913+0.4202
sampled_images = sampled_images * (max_tagert - min_tagert).item() + min_tagert.item()

sampled_images = einops.rearrange(sampled_images,"b c w h -> (b w) (c h)")
lable  = einops.rearrange(lable,"b c w h -> (b w) (c h)")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(sampled_images,cmap="jet")
ax1.set_title("Sampled Images")
ax1.axis("off")

# ax2.imshow(lable,cmap="jet",vmin = 0.0,vmax=10)
ax2.imshow(lable.cpu().detach().numpy().astype(np.uint8),cmap="jet")
ax2.set_title("Label")
ax2.axis("off")
plt.tight_layout()
plt.savefig("sampled_images_and_label.png")
