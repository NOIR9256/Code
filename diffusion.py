import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from tqdm import tqdm
import os
from diffusion import UNet_conditional
from utils import SpriteDataset, generate_animation
from train_TCIR import CustomIterableDataset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import math
from peft import LoraConfig, get_peft_model
from SRNdiff import train_sample
import einops

class DiffusionModel(nn.Module):
    def __init__(self, device=None,checkpoint_name=None,pre_model = None):
        super(DiffusionModel, self).__init__()
        self.device = device
        self.file_dir = os.path.dirname(__file__)
        self.checkpoint_name = checkpoint_name
        self.nn_model = self.initialize_nn_model(self.device,self.file_dir,self.checkpoint_name)
        self.model = self.loral_model(pre_model,self.device)
        self.ema_model = self.initialize_nn_model(self.device,self.file_dir,self.checkpoint_name)
        self.create_dirs(self.file_dir)


    # '''
    def train(self, batch_size=64, n_epoch=32, lr=1e-3, timesteps=500, beta1=1e-4, beta2=0.02,
              checkpoint_save_dir=None, image_save_dir=None, n_future_steps=8,schedule_name = None):
        """
        训练模型，使用预测的时刻作为下一步的条件

        Args:
            n_future_steps (int): 要预测的未来时刻数，默认为3
        """
        self.nn_model.train()
        # 噪声调度器:线性、余弦噪声
        _, _, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, schedule_name, self.device)
        # 数据加载
        sampler_file = None
        if sampler_file is None:
            sampler_file = {
                "train": "/home/dupf/code/ldcast-master/cache/train.pkl",
                "valid": "/home/dupf/code/ldcast-master/cache/valid.pkl",
                "test": "/home/dupf/code/ldcast-master/cache/test.pkl",
            }

        train_dataset = CustomIterableDataset(sampler_file['train'], batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True, pin_memory=True
        )
        # 数据归一化
        global_min = train_dataset.global_min
        global_max = train_dataset.global_max

        #优化器设置
        optim = self.initialize_optimizer(self.nn_model, lr, self.checkpoint_name, self.file_dir, self.device)
        scheduler = self.initialize_scheduler(optim, self.checkpoint_name, self.file_dir, self.device)

        loss_min = 0
        for epoch in range(self.get_start_epoch(self.checkpoint_name, self.file_dir),
                           self.get_start_epoch(self.checkpoint_name, self.file_dir) + n_epoch):
            ave_loss = 0
            for c, x in tqdm(train_loader, mininterval=2, desc=f"Epoch {epoch}"):
                # =================================================================================================================================================================
                x_or_list = torch.chunk(self.denormalize(x,global_min,global_max), chunks=n_future_steps, dim=1)
                x = x.to(self.device)
                c = c.to(self.device)
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps, (x.shape[0],)).to(self.device)

                # 前向扩散
                x_pert = self.perturb_input(x, t, noise, ab_t)
                x_noise_list = torch.chunk(self.denormalize(x_pert,global_min,global_max), chunks=n_future_steps, dim=1)

                # 预测噪声
                pred_noise = self.nn_model(x_pert, t , c)
                # 训练去除噪声观察结果
                x_denoised = self.get_x_unpert(x_pert, t, pred_noise, ab_t)
                x_denoised_list = torch.chunk(self.denormalize(x_denoised,global_min,global_max), chunks=n_future_steps, dim=1)
                # loss = torch.nn.functional.mse_loss(pred_noise, noise)

                # 改进的混合损失函数
                l1_loss = torch.nn.functional.l1_loss(x_denoised, x)
                # perceptual_loss = torch.nn.functional.l1_loss(pred_noise, noise)

                loss = torch.nn.functional.mse_loss(pred_noise, noise)

                # update params
                optim.zero_grad()
                loss.backward()
                optim.step()

                ave_loss += loss.item()/len(train_loader)

            scheduler.step()
            print(f"Epoch: {epoch}, loss: {ave_loss}")

            if (loss_min == 0 or ave_loss < loss_min):
                self.save_tensor_images(x_or_list, x_noise_list, x_denoised_list,
                                        epoch, self.file_dir, image_save_dir)
                # self.save_tensor_images1(x,x_pert, x_denoised,
                #                         epoch, self.file_dir, image_save_dir)
                self.save_checkpoint(self.nn_model, optim, scheduler, epoch, ave_loss,
                                     timesteps, beta1, beta2, self.device,
                                     train_loader.batch_size, self.file_dir, checkpoint_save_dir)
                # torch.save(self.nn_model.state_dict(), "/home/dupf/code/ldcast-master/conditional-ddpm/save_ckpt/finetuned.pth")
                loss_min = ave_loss
    # '''


    def denormalize(self,tensor, min_val, max_val):
        # return (tensor + 1) / 2 * (max_val - min_val) + min_val
        return (tensor * (max_val - min_val)) + min_val
    def denormalize_s_m(self,tensor, x_mean, x_std):
        return tensor * x_std + x_mean
    def denormalize_255(self,tensor):
        return ((tensor - tensor.min().item()) / (tensor.max().item() - tensor.min().item())) * 255

    @torch.no_grad()
    def sample_ddpm(self, n_samples, context=None, timesteps=None, 
                    beta1=None, beta2=None, labels =None, save_rate=20):
        """Returns the final denoised sample x0,
        intermediate samples xT, xT-1, ..., x1, and
        times tT, tT-1, ..., t1
        """
        # if all([timesteps, beta1, beta2]):
        #     a_t, b_t, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)
        # else:
        timesteps, a_t, b_t, ab_t = self.get_ddpm_params_from_checkpoint(self.file_dir,
                                                                    self.checkpoint_name,
                                                                    self.device,
                                                                    schedule_name = "cosine"
                                                                    #      linear cosine
                                                                    )

        self.nn_model.eval()

        samples = labels
        intermediate_samples = [samples.detach().cpu()] # samples at T = timesteps
        t_steps = [timesteps] # keep record of time to use in animation generation
        with torch.no_grad():
            for i in range(timesteps -1, 0, -1):
                print(f"Sampling timestep {i}", end="\r")
                if i % 50 == 0: print(f"Sampling timestep {i}")

                t = (torch.ones(n_samples) * i).long().to(self.device)

                z = torch.randn_like(samples) if i > 1 else 0
                pred_noise = self.nn_model(samples, t, context)
                samples = self.denoise_add_noise(samples, i, pred_noise, a_t, b_t, ab_t, z)

                if i % save_rate == 0 or i < 8:
                    intermediate_samples.append(samples.detach().cpu())
                    # intermediate_samples.append(samples.detach().cpu())
                    t_steps.append(i - 1)
        return intermediate_samples[-1], intermediate_samples, t_steps

    def perturb_input(self, x, t, noise, ab_t):
        """Perturbs given input
        i.e., Algorithm 1, step 5, argument of epsilon_theta in the article
        """
        # return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise
        return torch.sqrt(ab_t[t])[:, None, None, None] * x + (torch.sqrt(1. - ab_t[t])[:, None, None, None]) * noise

    def instantiate_dataset(self, dataset_name, transforms, file_dir, train=True):
        """Returns instantiated dataset for given dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10"}, "Unknown dataset"
        
        transform, target_transform = transforms
        if dataset_name=="mnist":
            return MNIST(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name=="fashion_mnist":
            return FashionMNIST(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name=="sprite":
            return SpriteDataset(os.path.join(file_dir, "datasets"), transform, target_transform)
        if dataset_name=="cifar10":
            return CIFAR10(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)

    def get_transforms(self, dataset_name):
        """Returns transform and target-transform for given dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10"}, "Unknown dataset"

        if dataset_name in {"mnist", "fashion_mnist", "cifar10"}:
            transform = transforms.Compose([
                transforms.ToTensor(),
                lambda x: 2*(x - 0.5)
            ])
            target_transform = transforms.Compose([
                lambda x: torch.tensor([x]),
                lambda class_labels, n_classes=10: nn.functional.one_hot(class_labels, n_classes).squeeze()
            ])

        if dataset_name=="sprite":
            transform = transforms.Compose([
                transforms.ToTensor(),  # from [0,255] to range [0.0,1.0]
                lambda x: 2*x - 1       # range [-1,1]
            ])
            target_transform = lambda x: torch.from_numpy(x).to(torch.float32)
        return transform, target_transform
    
    def get_x_unpert(self, x_pert, t, pred_noise, ab_t):
        """Removes predicted noise pred_noise from perturbed image x_pert"""
        # return (x_pert - (1 - ab_t[t, None, None, None]).sqrt() * pred_noise) / ab_t.sqrt()[t, None, None, None]
        return (x_pert - (torch.sqrt(1 - ab_t[t])[:, None, None, None]) * pred_noise) / torch.sqrt(ab_t[t])[:, None, None, None]

    def initialize_nn_model(self, device, file_dir , checkpoint_name):
        """Returns the instantiated model based on dataset name"""
        nn_model = UNet_conditional(c_in=4,c_out=4,device = device)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            nn_model.to(device)
            nn_model.load_state_dict(checkpoint["model_state_dict"])
            # nn_model.load_state_dict(checkpoint)
            return nn_model
        return nn_model.to(device)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, 
                        timesteps, beta1, beta2, device, batch_size,
                        file_dir, save_dir):
        """Saves checkpoint for given variables"""
        if save_dir is None:
            fpath = os.path.join(file_dir, "checkpoints", f"_checkpoint_{epoch}.pth")
        else:
            # fpath = os.path.join(save_dir, f"{dataset_name}_checkpoint_{epoch}.pth")
            fpath = os.path.join(save_dir, f"checkpoint_best-3.19-1623.pth")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "timesteps": timesteps, 
            "beta1": beta1, 
            "beta2": beta2,
            "device": device,
            # "dataset_name": dataset_name,
            "batch_size": batch_size
        }
        torch.save(checkpoint, fpath)
        # print()

    def create_dirs(self, file_dir):
        """Creates directories required for training"""
        dir_names = ["checkpoints", "saved-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(file_dir, dir_name), exist_ok=True)

    def initialize_optimizer(self, nn_model, lr, checkpoint_name, file_dir, device):
        """Instantiates and initializes the optimizer based on checkpoint availability"""
        optim = torch.optim.AdamW(nn_model.parameters(), lr=lr)
        # if checkpoint_name:
        #     checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
        #     optim.load_state_dict(checkpoint["optimizer_state_dict"])
        return optim

    def initialize_scheduler(self, optimizer, checkpoint_name, file_dir, device):
        """Instantiates and initializes scheduler based on checkpoint availability"""
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, 
                                                      end_factor=0.01, total_iters=2000)
        # if checkpoint_name:
        #     checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
        #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return scheduler
    
    def get_start_epoch(self, checkpoint_name, file_dir):
        """Returns starting epoch for training"""
        # if checkpoint_name:
        #     start_epoch = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name),
        #                             map_location=torch.device("cpu"))["epoch"] + 1
        # else:
        start_epoch = 0
        return start_epoch
    
    def save_tensor_images(self, x_orig,x_noise_per, x_denoised, cur_epoch, file_dir, save_dir):
        """
        将三组图像拼接成一张单通道大图并保存
        支持CUDA张量和复杂形状处理
        """
        if save_dir is None:
            fpath = os.path.join(file_dir, "saved-images-lora-3.18-2001", f"x_orig_noised_{cur_epoch}.jpeg")
            # fpath = os.path.join(file_dir, "saved-images", f"sample.jpeg")
        else:
            fpath = os.path.join(save_dir, f"x_orig_noised_denoised_{cur_epoch}.jpeg")
            # fpath = os.path.join(save_dir, f"x_orig_noised_denoised_best.jpeg")
        # '''
        # 确保张量在CPU上
        x_list = [step.cpu() for step in x_orig]
        x_noise_steps = [step.cpu() for step in x_noise_per]
        # x_noised = x_noised.cpu()
        # x_denoised = x_denoised.cpu()
        x_denoised_steps = [step.cpu() for step in x_denoised]

        # 处理图像组
        # images = x_list + x_noise_steps + x_denoised_steps
        images = x_denoised_steps
        # normalized_images = [normalize_image(img.detach()) for img in images]
        normalized_images = [img.detach() for img in images]

        # 创建网格布局
        grid_images = []
        for i in range(len(normalized_images[0])):  # 遍历batch中的每个图像
            row_images = [img[i].squeeze() for img in normalized_images]  # 取每组图像的第i个
            row = torch.cat(row_images, dim=1)  # 水平拼接
            grid_images.append(row)


        final_image = torch.cat(grid_images, dim=0)  # 垂直拼接


        plt.figure(figsize=(20, 10))
        plt.imshow(final_image.numpy().astype(np.uint8), cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0,dpi=300)
        plt.close()
        # '''


    def get_ddpm_noise_schedule(self, timesteps, beta1, beta2, schedule_name,device,s=0.008):
        """Returns ddpm noise schedule variables, a_t, b_t, ab_t
        b_t: \beta_t
        a_t: \alpha_t
        ab_t \bar{\alpha}_t
        """
        if schedule_name == "linear":

            beta = torch.linspace(beta1, beta2, timesteps + 1, device=device)
            alpha = 1 - beta
            alpha_hat = torch.cumprod(alpha, dim=0)
            return alpha, beta, alpha_hat
        elif schedule_name == "cosine":

            t = torch.arange(timesteps + 1, dtype=torch.float32, device=device)
            # alpha_bar_t (Improved DDPM)
            alpha_bar_t = torch.cos((t / timesteps + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar_t = alpha_bar_t / alpha_bar_t[0]
            alpha_t = alpha_bar_t[1:] / alpha_bar_t[:-1]
            beta = 1 - alpha_t
            beta = torch.clip(beta, 0.0001, 0.9999)


            alpha = 1. - beta
            alpha_hat = torch.cumprod(alpha, dim=0)

            return  alpha, beta, alpha_hat
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    
    def get_ddpm_params_from_checkpoint(self, file_dir, checkpoint_name,device,schedule_name ):
        """Returns scheduler variables T, a_t, ab_t, and b_t from checkpoint"""
        checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), torch.device("cpu"))
        T = checkpoint["timesteps"]
        a_t, b_t, ab_t = self.get_ddpm_noise_schedule(T, checkpoint["beta1"], checkpoint["beta2"],schedule_name, device)
        return T,a_t ,b_t , ab_t
    
    def denoise_add_noise(self, x, t, pred_noise, a_t, b_t, ab_t, z):
        """Removes predicted noise from x and adds gaussian noise z
        i.e., Algorithm 2, step 4 at the ddpm article
        """
        noise = b_t.sqrt()[t] * z
        denoised_x = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
        return denoised_x + noise

    def initialize_dataset_name(self, file_dir, checkpoint_name, dataset_name):
        """Initializes dataset name based on checkpoint availability"""
        if checkpoint_name:
            return torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["dataset_name"]
        return dataset_name
    
    def initialize_dataloader(self, dataset, batch_size, checkpoint_name, file_dir):
        """Returns dataloader based on batch-size of checkpoint if present"""
        if checkpoint_name:
            batch_size = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["batch_size"]
        return DataLoader(dataset, batch_size, True)
    
    def get_masked_context(self, context, p=0.9):
        "Randomly mask out context"
        return context*torch.bernoulli(torch.ones((context.shape[0], 1))*p)
    
    def save_generated_samples_into_folder(self, n_samples, context, folder_path, **kwargs):
        """Save DDPM generated inputs into a specified directory"""
        samples, _, _ = self.sample_ddpm(n_samples, context, **kwargs)
        for i, sample in enumerate(samples):
            save_image(sample, os.path.join(folder_path, f"image_{i}.jpeg"))
    
    def save_dataset_test_images(self, n_samples):
        """Save dataset test images with specified number"""
        folder_path = os.path.join(self.file_dir, f"{self.dataset_name}-test-images")
        os.makedirs(folder_path, exist_ok=True)

        dataset = self.instantiate_dataset(self.dataset_name, 
                            (transforms.ToTensor(), None), self.file_dir, train=False)
        dataloader = DataLoader(dataset, 1, True)
        for i, (image, _) in enumerate(dataloader):
            if i == n_samples: break
            save_image(image, os.path.join(folder_path, f"image_{i}.jpeg"))

    
    def get_custom_context(self, n_samples, n_classes, device):
        """Returns custom context in one-hot encoded form"""
        context = []
        for i in range(n_classes - 1):
            context.extend([i]*(n_samples//n_classes))
        context.extend([n_classes - 1]*(n_samples - len(context)))
        return torch.nn.functional.one_hot(torch.tensor(context), n_classes).float().to(device)
    
    def generate(self, n_samples, n_images_per_row, timesteps, beta1, beta2):
        """Generates x0 and intermediate samples xi via DDPM, 
        and saves as jpeg and gif files for given inputs
        """
        root = os.path.join(self.file_dir, "generated-images")
        os.makedirs(root, exist_ok=True)
        condition, min , max,labels= self.get_ty_custom_context(n_samples, self.device)
        x0, intermediate_samples, t_steps = self.sample_ddpm(n_samples,
                                                             condition,  # 直接传递台风数据作为上下文
                                                             timesteps,
                                                             beta1,
                                                             beta2,labels)
        x0 = self.denormalize_255(x0.to(self.device))
        torch.save(x0, '/home/dupf/code/ldcast-master/conditional-ddpm/generated-images/generated_3-19.pt')
        x0 = x0.reshape(-1, 1, 256, 256)
        save_image(x0, os.path.join(root, f"TCIR_ddpm_images.jpeg"), nrow=n_samples)
        # generate_animation(intermediate_samples,
        #                    t_steps,
        #                    os.path.join(root, f"{self.dataset_name}_ani.gif"),
        #                    n_images_per_row)

    def get_ty_custom_context(self, n_samples, device):
        """Returns custom context as the first 8 frames of typhoon images"""

        sampler_file = None
        if sampler_file is None:
            sampler_file = {
                "train": "../cache/IR_train.pkl",
                "valid": "../cache/IR_valid.pkl",
                "test": "../cache/IR_test.pkl",
            }

        train_dataset = CustomIterableDataset(sampler_file['train'], n_samples)
        train_loader = DataLoader(
            train_dataset,
            batch_size=n_samples,
            shuffle=False,
            drop_last=True, pin_memory=True
        )

        # test_dataset = CustomIterableDataset(sampler_file['test'], n_samples)
        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=n_samples,
        #     shuffle=False,
        #     drop_last=True, pin_memory=True
        # )
        # global_min = train_dataset.global_min
        # global_max = train_dataset.global_max

        for features, labels in train_loader:
            condition = features.to(device)  # Get condition
            labels = labels.to(device)
            break

        # 保存目标原始矩阵、条件矩阵
        torch.save(labels, '/home/dupf/code/ldcast-master/conditional-ddpm/generated-images/orginal.pt')
        torch.save(condition, '/home/dupf/code/ldcast-master/conditional-ddpm/generated-images/orginal_condition.pt')

        # context = self.typhoon_data[:, :, :, :, :].to(device)  # 取前 8 个时间步
        # condition = condition.repeat(n_samples, 1, 1, 1)  # 复制 n_samples 次
        # labels = labels.repeat(n_samples, 1, 1, 1)  # 复制 n_samples 次

        # 数据反归一化
        min_val, max_val = condition.min().to(device), condition.max().to(device)
        min_lab, max_lab = labels.min().to(device), labels.max().to(device)

        condition = ((condition - min_val) / (max_val - min_val)).to(self.device)
        labels = ((labels - min_lab) / (max_lab - min_lab)).to(self.device)

        # condition = self.denormalize(condition,global_min,global_max)
        # labels = self.denormalize(labels,global_min,global_max)

        return condition, min_val, max_val,labels
