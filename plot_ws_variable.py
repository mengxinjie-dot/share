import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc  # 添加垃圾回收模块
import numpy.ma as ma  # 添加masked array支持

# 创建picture文件夹
os.makedirs('picture', exist_ok=True)

# 读取NetCDF文件
file_path = 'tongren02_OUTPUT/tongren02_3d.000.nc'
data = nc.Dataset(file_path)

# 只读取需要的时间步长和高度层数据
selected_levels = [2, 3, 4, 5, 6, 9] # 13, 19, 26, 40, 55, 70
num_levels = len(selected_levels)
num_times = 10

# 获取时间和高度信息
times = data.variables['time'][:]
z = data.variables['zu_3d'][selected_levels]  # 只读取选定的高度层
selected_times = np.linspace(0, len(times)-1, num_times, dtype=int)

# 创建自定义colormap
from matplotlib.colors import LinearSegmentedColormap
colors = ['#ffffff', '#add8e6', '#4169e1', '#000080']  # 白色到浅蓝到深蓝
cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

# 创建大图
fig, axes = plt.subplots(num_levels, num_times, 
                        figsize=(2*num_times, 2*num_levels),
                        sharex=True, sharey=True)

# 调整子图之间的间距
plt.subplots_adjust(left=0.1, right=0.95, 
                   bottom=0.1, top=0.95,  # 增加底部边距，为colorbar留出空间
                   wspace=0.05, hspace=0.1)

# 预分配存储空间
wind_data = {}  # 使用字典存储计算结果
wind_speed_min = float('inf')
wind_speed_max = float('-inf')

# 计算并存储所有需要的风速数据
total_iterations = len(selected_levels) * len(selected_times)
progress_bar = tqdm(total=total_iterations, desc="计算风速")

# 设置采样间隔（每隔多少个网格点画一个箭头）
sample_step = 10

for i, level in enumerate(selected_levels):
    for j, time_idx in enumerate(selected_times):
        # 分块读取风速数据
        u = ma.masked_invalid(data.variables['u'][time_idx:time_idx+1, level][0])  # 去除多余的维度并处理无效值
        v = ma.masked_invalid(data.variables['v'][time_idx:time_idx+1, level][0])  # 去除多余的维度并处理无效值
        
        # 使用masked array计算风速，这样会自动处理无效值
        wind_speed = ma.sqrt(u**2 + v**2)
        
        # 更新最大最小值（使用压缩后的数据以忽略掩码值）
        valid_speeds = wind_speed.compressed()
        if len(valid_speeds) > 0:  # 确保有有效数据
            wind_speed_min = min(wind_speed_min, np.min(valid_speeds))
            wind_speed_max = max(wind_speed_max, np.max(valid_speeds))
        
        # 存储结果
        key = (i, j)
        wind_data[key] = {
            'speed': wind_speed,
            'u': u,
            'v': v
        }
        
        # 清理临时变量
        del u, v, valid_speeds
        progress_bar.update(1)

progress_bar.close()

# 绘制图像
progress_bar = tqdm(total=total_iterations, desc="绘制进度")

for i in range(num_levels):
    for j in range(num_times):
        ax = axes[i, j]
        wind_speed = wind_data[(i, j)]['speed']
        u = wind_data[(i, j)]['u']
        v = wind_data[(i, j)]['v']
        
        # 调整vmin和vmax，风速只使用正值
        im = ax.imshow(wind_speed, cmap=cmap, origin='lower',
                      vmin=0, vmax=wind_speed_max)
        
        # 添加风场箭头（只在有效值的位置）
        y, x = np.mgrid[0:wind_speed.shape[0]:sample_step, 0:wind_speed.shape[1]:sample_step]
        # 只在非掩码位置绘制箭头
        mask = ~(u[::sample_step, ::sample_step].mask | v[::sample_step, ::sample_step].mask)
        ax.quiver(x[mask], y[mask], 
                 u[::sample_step, ::sample_step][mask],
                 v[::sample_step, ::sample_step][mask],
                 color='black', scale=50, width=0.002)
        
        # 设置标题和标签
        if i == 0:
            ax.set_title(f't = {times[selected_times[j]]:.2f} s')
        if j == 0:
            ax.set_ylabel(f'z = {z[i]:.2f} m')
        
        progress_bar.update(1)

# 清理内存
del wind_data
gc.collect()

# 关闭进度条
progress_bar.close()

# 添加共享的colorbar
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('wind speed (m/s)')

# 保存图片
plt.savefig('picture/wind_speed_all_levels_times.png', dpi=600, bbox_inches='tight')
plt.close()

# 关闭数据文件
data.close()

print("成功保存所有高度和时间步长的组合图片到picture/wind_speed_all_levels_times.png")
