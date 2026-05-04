import torch

path = "conditional_ddpm/SRNdiff/5-20-label.pth"
data = torch.load(path)

print("\n=== 元数据 ===")
print(f"文件版本: {torch.__version__}")  # 显示使用的PyTorch版本
print(f"数据类型: {type(data)}")

print("\n=== 内容结构 ===")
print(f"Keys: {list(data.keys())}")
print(f"'tagert'形状: {data['tagert'].shape} (设备: {data['tagert'].device})")
print(f"'label'形状: {data['label'].shape} (设备: {data['label'].device})")

print("\n=== 数值范围 ===")
print(f"tagert - 最小值: {data['tagert'].min().item():.4f} 最大值: {data['tagert'].max().item():.4f}")
print(f"label - 最小值: {data['label'].min().item():.4f} 最大值: {data['label'].max().item():.4f}")