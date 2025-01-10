import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
import multiprocessing as mp
from functools import partial

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

def create_3d_wind_animation(file_path, output_path, mode='bottom', n_layers=3, sample_rate=1):
    """创建3D风场动画"""
    # 读取数据集
    dataset = nc.Dataset(file_path)
    
    # 创建图形和3D轴
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取第一帧数据来设置图形参数
    x, y, z, wind_speed = process_wind_data(dataset, 0, sample_rate, n_layers, mode)
    
    # 创建网格点
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # 计算坐标范围
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z[:n_layers]) - np.min(z[:n_layers])
    
    # 设置视角和比例
    ax.view_init(elev=30, azim=290)
    ax.set_box_aspect([x_range/x_range, y_range/x_range, z_range/x_range])
    
    # 初始化散点图（使用空数据）
    scatter = ax.scatter([], [], [], c=[], cmap='jet', s=0.5, alpha=0.5)
    
    # 添加颜色条
    plt.colorbar(scatter, ax=ax, label='Wind Speed (m/s)')
    
    # 设置标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 添加指北箭头
    arrow_length = x_range * 0.1
    arrow_x = np.max(x) - arrow_length * 0.5
    arrow_y = np.max(y) - arrow_length * 0.5
    arrow_z = np.max(z[:n_layers]) * 1.2
    
    ax.quiver(arrow_x, arrow_y, arrow_z, 0, arrow_length, 0,
             color='red', arrow_length_ratio=0.2, linewidth=2)
    ax.quiver(arrow_x, arrow_y, arrow_z, arrow_length, 0, 0,
             color='blue', arrow_length_ratio=0.2, linewidth=2)
    
    ax.text(arrow_x + arrow_length * 1.2, arrow_y, arrow_z, 'E', color='blue', fontsize=16, fontweight='bold')
    ax.text(arrow_x, arrow_y + arrow_length * 1.2, arrow_z, 'N', color='red', fontsize=16, fontweight='bold')
    
    def update(frame):
        """更新动画帧"""
        ax.cla()  # 清除当前帧
        
        # 重新处理数据
        x, y, z, wind_speed = process_wind_data(dataset, frame, sample_rate, n_layers, mode)
        
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
        
        points = np.array(points)
        colors_array = np.array(colors_array)
        
        # 绘制散点图
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=colors_array, cmap='jet', s=0.5, alpha=0.5)
        
        # 重新设置视角和标签
        ax.view_init(elev=30, azim=290)
        ax.set_box_aspect([x_range/x_range, y_range/x_range, z_range/x_range])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # 重新绘制指北箭头
        ax.quiver(arrow_x, arrow_y, arrow_z, 0, arrow_length, 0,
                 color='red', arrow_length_ratio=0.2, linewidth=2)
        ax.quiver(arrow_x, arrow_y, arrow_z, arrow_length, 0, 0,
                 color='blue', arrow_length_ratio=0.2, linewidth=2)
        ax.text(arrow_x + arrow_length * 1.2, arrow_y, arrow_z, 'E', 
                color='blue', fontsize=16, fontweight='bold')
        ax.text(arrow_x, arrow_y + arrow_length * 1.2, arrow_z, 'N', 
                color='red', fontsize=16, fontweight='bold')
        
        # 更新标题
        plt.title(f'Wind Speed Distribution (Time Step {frame})')
        
        return scatter,
    
    # 创建动画
    frames = range(0, 200, 1)  # 每10个时间步取一帧，总共200帧
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                 interval=200, blit=False)
    
    # 保存动画
    writer = animation.PillowWriter(fps=5)
    anim.save(output_path, writer=writer, dpi=300)
    
    # 关闭数据集和图形
    dataset.close()
    plt.close()

def process_animation(params, file_path):
    """单个动画处理函数"""
    output_path = f'animations/wind_speed_3d_{params["mode"]}_{params["n_layers"]}layers_sr{params["sample_rate"]}.gif'
    print(f"生成动画: {output_path}")
    create_3d_wind_animation(
        file_path, 
        output_path, 
        mode=params['mode'], 
        n_layers=params['n_layers'], 
        sample_rate=params['sample_rate']
    )

def main():
    # 创建输出目录
    os.makedirs('animations', exist_ok=True)
    
    # 设置基础参数
    file_path = '/home/mxj/docker_palm/share/tongren02_OUTPUT/tongren02_3d.000.nc'
    
    # 定义不同参数组合
    param_combinations = [
        {'mode': 'bottom', 'n_layers': 4, 'sample_rate': 2},
        {'mode': 'bottom', 'n_layers': 5, 'sample_rate': 2},
        {'mode': 'terrain', 'n_layers': 1, 'sample_rate': 2},
        {'mode': 'terrain', 'n_layers': 3, 'sample_rate': 2},
        {'mode': 'terrain', 'n_layers': 5, 'sample_rate': 2},
        {'mode': 'terrain', 'n_layers': 10, 'sample_rate': 2},
    ]
    
    print("开始并行生成风场动画...")
    
    # 创建进程池
    with mp.Pool() as pool:
        # 使用partial固定file_path参数
        process_func = partial(process_animation, file_path=file_path)
        # 并行处理所有参数组合
        pool.map(process_func, param_combinations)
    
    print("所有动画生成完成！")

if __name__ == "__main__":
    main() 