import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import os
from itertools import product
import multiprocessing as mp
from tqdm import tqdm

def create_3d_wind_plot(x, y, z, wind_speed, mode, n_layers, time_step, output_path):
    """创建3D风场图"""
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 设置自定义颜色映射
    custom_cmap = plt.cm.jet  # 使用matplotlib内置的jet颜色映射

    # 创建网格点
    xx, yy = np.meshgrid(x, y, indexing='ij')
    points = []
    colors_array = []

    # 收集所有点和对应的颜色
    if mode == 'bottom':
        for k in range(n_layers):
            mask = ~np.isnan(wind_speed[:, :, k])
            points.extend(zip(xx[mask].flatten(), yy[mask].flatten(), np.full_like(xx[mask].flatten(), z[k])))
            colors_array.extend(wind_speed[:, :, k][mask].flatten())
    else:
        for k in range(len(z)):
            mask = ~np.isnan(wind_speed[:, :, k])
            points.extend(zip(xx[mask].flatten(), yy[mask].flatten(), np.full_like(xx[mask].flatten(), z[k])))
            colors_array.extend(wind_speed[:, :, k][mask].flatten())

    # 转换为numpy数组
    points = np.array(points)
    colors_array = np.array(colors_array)

    # 计算坐标范围
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)

    # 根据模式选择合适的z范围计算方法
    if mode == 'bottom':
        z_range = np.max(z[:n_layers]) - np.min(z[:n_layers])
    else:
        valid_z_indices = np.unique(np.where(~np.isnan(wind_speed))[2])
        if len(valid_z_indices) > 0:
            max_z_index = np.max(valid_z_indices)
            min_z_index = np.min(valid_z_indices)
            z_range = z[max_z_index] - z[min_z_index]
        else:
            z_range = np.max(z[:n_layers]) - np.min(z[:n_layers])

    # 绘制散点图
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=colors_array,
                        cmap=custom_cmap,
                        s=0.5,
                        alpha=0.5)

    # 设置视角
    ax.view_init(elev=30, azim=290)

    # 添加指北箭头
    # 计算箭头位置（移到图的右上角）
    arrow_length = x_range * 0.1  # 箭头长度为x范围的15%
    arrow_x = np.max(x) - arrow_length * 0.5  # 更靠近右边
    arrow_y = np.max(y) - arrow_length * 0.5  # 更靠近上边
    if mode == 'bottom':
        arrow_z = np.max(z[:n_layers]) * 1.2
    else:
        arrow_z = np.max(z[:max_z_index]) * 1.2
    
    # 添加指北箭头（红色）
    ax.quiver(arrow_x, arrow_y, arrow_z,  # 起点
              0, arrow_length, 0,          # 方向向量（y轴正方向为北）
              color='red', 
              arrow_length_ratio=0.2,
              linewidth=2)
    
    # 添加指东箭头（蓝色）
    ax.quiver(arrow_x, arrow_y, arrow_z,  # 起点
              arrow_length, 0, 0,          # 方向向量（x轴正方向为东）
              color='blue', 
              arrow_length_ratio=0.2,
              linewidth=2)
    
    # 添加方向标签
    ax.text(arrow_x + arrow_length * 1.2, arrow_y, arrow_z, 'E', color='blue', fontsize=16, fontweight='bold')
    ax.text(arrow_x, arrow_y + arrow_length * 1.2, arrow_z, 'N', color='red', fontsize=16, fontweight='bold')

    # 设置坐标轴比例相等
    ax.set_box_aspect([x_range/x_range, y_range/x_range, z_range/x_range])

    # 调整图形布局，确保所有标签都可见
    plt.tight_layout(pad=2.0)  # 增加padding

    # 设置坐标轴标签的字体大小和位置
    ax.set_xlabel('X (m)', fontsize=12, labelpad=15)
    ax.set_ylabel('Y (m)', fontsize=12, labelpad=15)
    ax.set_zlabel('Z (m)', fontsize=12, labelpad=15)

    # 设置z轴刻度和标签
    z_ticks = ax.get_zticks()
    ax.set_zticks(z_ticks[::2])  # 减少刻度数量

    # 调整z轴标签的位置和旋转
    ax.tick_params(axis='z', pad=5)  # 增加刻度标签与轴的距离

    # 添加颜色条和标签
    cbar = fig.colorbar(scatter, ax=ax, label='Wind Speed (m/s)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 设置标题
    mode_text = 'Bottom' if mode == 'bottom' else 'Terrain'
    plt.title(f'Wind Speed Distribution ({mode_text} {n_layers} Layers, Time Step {time_step})')

    # 保存图片
    plt.savefig(output_path, dpi=600)
    plt.close('all')

def process_wind_data(dataset, time_step, sample_rate, n_layers, mode='bottom'):
    """处理风速数据"""
    # 获取网格坐标并采样
    x = dataset.variables['x'][::sample_rate]
    y = dataset.variables['y'][::sample_rate]
    z = dataset.variables['zu_3d'][::sample_rate]
    
    # 获取指定时间步的数据并采样
    u_data = dataset.variables['u'][time_step, ::sample_rate, ::sample_rate, ::sample_rate]
    v_data = dataset.variables['v'][time_step, ::sample_rate, ::sample_rate, ::sample_rate]
    
    # 计算风速大小
    wind_speed = np.sqrt(np.clip(u_data**2 + v_data**2, 0, None))
    
    if mode == 'bottom':
        wind_speed = wind_speed[:n_layers, :, :]
    else:
        mask = ~np.ma.getmask(wind_speed)
        terrain_wind_speed = np.full_like(wind_speed, np.nan)
        
        for i in range(wind_speed.shape[1]):
            for j in range(wind_speed.shape[2]):
                valid_levels = np.where(mask[:, i, j])[0]
                if len(valid_levels) > 0:
                    start_level = valid_levels[0]
                    end_level = min(start_level + n_layers, wind_speed.shape[0])
                    terrain_wind_speed[start_level:end_level, i, j] = wind_speed[start_level:end_level, i, j]
        
        wind_speed = terrain_wind_speed
    
    wind_speed = np.transpose(wind_speed, (2, 1, 0))
    return x[:wind_speed.shape[0]], y[:wind_speed.shape[1]], z, wind_speed

def process_combination(args):
    """处理单个参数组合"""
    time_step, n_layers, mode, file_path = args
    
    # 读取nc文件
    dataset = nc.Dataset(file_path)
    
    # 处理数据
    x, y, z, wind_speed = process_wind_data(dataset, time_step, 1, n_layers, mode)
    
    # 生成输出文件名
    output_path = f'picture/wind_speed_3d_{mode}_{time_step}_{n_layers}.png'
    
    # 创建并保存图像
    create_3d_wind_plot(x, y, z, wind_speed, mode, n_layers, time_step, output_path)
    
    # 关闭数据集
    dataset.close()
    
    return output_path

def main():
    # 创建picture文件夹
    os.makedirs('picture', exist_ok=True)

    # 设置参数范围
    time_steps = [50, 100, 150, 200]  # 时间步列表
    # n_layers_list = [5, 10, 15]  # 层数列表
    n_layers_list = [3]  # 层数列表
    # modes = ['bottom', 'terrain']  # 模式列表
    modes = ['bottom']  # 模式列表
    file_path = '/home/mxj/docker_palm/share/tongren02_OUTPUT/tongren02_3d.000.nc'

    # 创建参数组合
    combinations = [(t, n, m, file_path) for t, n, m in product(time_steps, n_layers_list, modes)]
    total_combinations = len(combinations)

    print(f"\n开始处理 {total_combinations} 个参数组合...")
    
    # 获取CPU核心数
    num_cores = mp.cpu_count()
    print(f"使用 {num_cores} 个CPU核心进行并行处理")

    # 创建进程池
    with mp.Pool(num_cores) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(process_combination, combinations),
            total=total_combinations,
            desc="处理进度"
        ))

    print("\n所有组合处理完成！")
    print("生成的图片文件：")
    for result in results:
        print(f"- {result}")

if __name__ == "__main__":
    main() 