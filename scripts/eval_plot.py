import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import properscoring as ps
import seaborn as sns

# 读取nc文件
nc_file = '/home/dupf/ldcast/ldcast-master/results/eval_ensembles/TCIR/ensemble_batch-0000.nc'
data = xr.open_dataset(nc_file)

# 提取预测和真实结果
forecast = data['forecasts'].values
future_observations = data['future_observations'].values

# 初始化存储结果的数组
rmse = np.zeros(forecast.shape[2])
mse = np.zeros(forecast.shape[2])
crps = np.zeros(forecast.shape[2])

# 计算RMSE、MSE、CRPS
for t in range(forecast.shape[2]):
    forecast_t = forecast[:, :, t, :, :, :]  # shape: (8, 1, 256, 256, 32)
    future_observations_t = future_observations[:, :, t, :, :]  # shape: (8, 1, 256, 256)

    # 按成员平均计算RMSE和MSE
    mse_t = np.mean((forecast_t - future_observations_t[..., None])**2, axis=(0, 1, 2, 3))
    rmse_t = np.sqrt(mse_t)

    # 计算CRPS
    crps_members = np.zeros(forecast.shape[-1])
    for m in range(forecast.shape[-1]):
        forecast_member = forecast_t[..., m].reshape(-1)
        crps_members[m] = ps.crps_ensemble(future_observations_t.reshape(-1), forecast_member[:, np.newaxis]).mean()

    mse[t] = mse_t.mean()
    rmse[t] = rmse_t.mean()
    crps[t] = crps_members.mean()

# 绘制曲线图
time_points = np.arange(1, forecast.shape[2] + 1)

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(time_points, rmse, marker='o', label='RMSE')
plt.title('RMSE over time')
plt.xlabel('Time')
plt.ylabel('RMSE')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_points, mse, marker='o', label='MSE')
plt.title('MSE over time')
plt.xlabel('Time')
plt.ylabel('MSE')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_points, crps, marker='o', label='CRPS')
plt.title('CRPS over time')
plt.xlabel('Time')
plt.ylabel('CRPS')
plt.legend()

plt.tight_layout()
plt.show()
