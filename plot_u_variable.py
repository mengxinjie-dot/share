import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 创建picture文件夹
os.makedirs('picture', exist_ok=True)

# 读取NetCDF文件
file_path = 'tongren02_OUTPUT/tongren02_3d.000.nc'
data = nc.Dataset(file_path)

# 获取u变量和高度信息
u = data.variables['u'][:]
print(type(u))
z = data.variables['zu_3d'][:]
times = data.variables['time'][:]

# 选择10个高度层和所有时间步长
num_levels = 12
selected_levels = [2, 3, 4, 5, 6, 9, 13, 19, 26, 40, 55, 70]  # 主要选择低高度
num_times = 20
selected_times = np.linspace(0, len(times)-1, num_times, dtype=int)

# 创建自定义colormap
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝色 -> 白色 -> 红色
cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

# 创建大图
fig, axes = plt.subplots(num_levels, num_times, 
                        figsize=(2*num_times, 2*num_levels),
                        sharex=True, sharey=True)

# 获取u变量的最大最小值用于统一colorbar
u_min = np.min(u)
u_max = np.max(u)

# 计算总迭代次数
total_iterations = len(selected_levels) * len(selected_times)

# 创建进度条
progress_bar = tqdm(total=total_iterations, desc="绘制进度")

# 绘制每个高度层和时间步长的平面图
for i, level in enumerate(selected_levels):
    for j, time in enumerate(selected_times):
        ax = axes[i, j]
        # 调整vmin和vmax使0值对应白色
        abs_max = max(abs(u_min), abs(u_max))
        im = ax.imshow(u[time, level], cmap=cmap, origin='lower',
                      vmin=-abs_max, vmax=abs_max)
        
        # 设置标题和标签
        if i == 0:
            ax.set_title(f't = {times[time]:.2f} s')
        if j == 0:
            ax.set_ylabel(f'z = {z[level]:.2f} m')
            
        # 更新进度条
        progress_bar.update(1)

# 关闭进度条
progress_bar.close()

# 添加共享的colorbar
cbar = fig.colorbar(im, ax=axes[-1, :].tolist(),
                   orientation='horizontal', fraction=0.2, pad=0.1,
                   location='bottom')

cbar.set_label('u (m/s)')

# 调整布局并保存图片
plt.subplots_adjust(bottom=0.15)  # 为colorbar留出空间
plt.savefig('picture/u_all_levels_times.png', dpi=300)
plt.close()

print("成功保存所有高度和时间步长的组合图片到picture/u_all_levels_times.png")
