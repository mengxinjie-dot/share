import netCDF4 as nc
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from wind_visualization_methods import WindVisualizer
from itertools import product

def process_wind_data(dataset, time_step, sample_rate, n_layers, mode='bottom'):
    """处理风速数据"""
    # 获取网格坐标并采样
    sample_rate_z = 1
    x = dataset.variables['x'][::sample_rate]
    y = dataset.variables['y'][::sample_rate]
    z = dataset.variables['zu_3d'][::sample_rate_z]
    
    # 获取指定时间步的数据并采样
    u_data = dataset.variables['u'][time_step, ::sample_rate_z, ::sample_rate, ::sample_rate]
    v_data = dataset.variables['v'][time_step, ::sample_rate_z, ::sample_rate, ::sample_rate]
    w_data = dataset.variables['w'][time_step, ::sample_rate_z, ::sample_rate, ::sample_rate] if 'w' in dataset.variables else None
    
    # 计算风速大小
    wind_speed = np.sqrt(np.clip(u_data**2 + v_data**2 + w_data**2, 0, None))

    if mode == 'bottom':
        u_data = u_data[:n_layers, :, :]
        v_data = v_data[:n_layers, :, :]
        if w_data is not None:
            w_data = w_data[:n_layers, :, :]
        wind_speed = wind_speed[:n_layers, :, :]
    else:
        # 创建掩码数组
        mask = ~np.ma.getmask(u_data)
        
        # 创建地形跟随的数据数组
        terrain_u = np.ma.masked_all(u_data.shape)
        terrain_v = np.ma.masked_all(v_data.shape) 
        terrain_w = np.ma.masked_all(w_data.shape) if w_data is not None else None
        terrain_speed = np.ma.masked_all(wind_speed.shape)
        
        # 对每个水平网格点进行处理
        for i, j in np.ndindex(u_data.shape[1:]):
            valid_levels = np.where(mask[:, i, j])[0]
            if len(valid_levels) > 0:
                start_level = valid_levels[0]
                end_level = min(start_level + n_layers, u_data.shape[0])
                
                # 复制有效数据
                terrain_u[start_level:end_level, i, j] = u_data[start_level:end_level, i, j]
                terrain_v[start_level:end_level, i, j] = v_data[start_level:end_level, i, j]
                terrain_speed[start_level:end_level, i, j] = wind_speed[start_level:end_level, i, j]
                if w_data is not None:
                    terrain_w[start_level:end_level, i, j] = w_data[start_level:end_level, i, j]
        
        # 更新数据
        u_data = terrain_u
        v_data = terrain_v
        w_data = terrain_w
        wind_speed = terrain_speed
    
    u_data = np.transpose(u_data, (2, 1, 0))
    v_data = np.transpose(v_data, (2, 1, 0))
    if w_data is not None:
        w_data = np.transpose(w_data, (2, 1, 0))
    wind_speed = np.transpose(wind_speed, (2, 1, 0))
    return x[:u_data.shape[0]], y[:u_data.shape[1]], z, u_data, v_data, w_data, wind_speed

def process_combination(args):
    """处理单个参数组合"""
    time_step, n_layers, mode, plot_types, display_topo, file_path = args
    
    # 读取nc文件
    dataset = nc.Dataset(file_path)
    
    # 处理数据
    x, y, z, u, v, w, wind_speed = process_wind_data(dataset, time_step, 2, n_layers, mode)
    
    # 创建可视化器
    visualizer = WindVisualizer(mode,x, y, z, u, v, w, wind_speed)
    
    # 创建输出目录
    os.makedirs('picture', exist_ok=True)
    os.makedirs('animations', exist_ok=True)
    
    # 生成不同类型的可视化
    base_name = f'wind_speed_3d_{mode}_{time_step}_{n_layers}'
    
    # 根据选择的类型生成图像
    if 'scatter' in plot_types:
        visualizer.scatter_plot(f'picture/{base_name}_scatter.png', n_layers, display_topo)
    
    if 'streamline' in plot_types:
        visualizer.streamline_plot(f'picture/{base_name}_streamline.png')
    
    if 'quiver' in plot_types:
        visualizer.quiver_plot(f'picture/{base_name}_quiver.png')
    
    if 'isosurface' in plot_types:
        visualizer.isosurface_plot(f'picture/{base_name}_isosurface.png')
    
    if 'volume' in plot_types:
        visualizer.volume_render_plot(f'picture/{base_name}_volume.png')
    
    if 'slice' in plot_types:
        visualizer.slice_plot(f'picture/{base_name}_slice.png')
    
    if 'particle' in plot_types:
        visualizer.particle_animation(f'animations/{base_name}_particles.gif')
    
    # 关闭数据集
    dataset.close()
    
    return base_name

def main():
    # 是否使用并行计算
    use_parallel = False
    # 设置参数范围
    time_steps = [50]  # 时间步列表
    n_layers_list = [5]  # 层数列表
    modes = ['terrain']  # 模式列表
    # 可选的图像类型：'scatter', 'streamline', 'quiver', 'isosurface', 'volume', 'slice', 'particle' 分别是散点图，流线图，箭头场，等值面，体渲染，切片图，粒子动画
    plot_types = ['scatter']  # 这里可以选择要生成的图像类型
    if 'scatter' in plot_types:
        display_topo = True
    else:
        display_topo = False
    file_path = '/home/mxj/docker_palm/share/tongren02_OUTPUT/tongren02_3d.000.nc'

    # 创建参数组合
    combinations = [(t, n, m, plot_types, display_topo, file_path) 
                   for t, n, m in product(time_steps, n_layers_list, modes)]
    total_combinations = len(combinations)

    print(f"\n开始处理 {total_combinations} 个参数组合...")
    print(f"将生成以下类型的图像: {', '.join(plot_types)}")
    
    if use_parallel:
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
    else:
        # 串行处理
        print("使用串行处理")
        results = []
        for combination in tqdm(combinations, desc="处理进度"):
            result = process_combination(combination)
            results.append(result)

    print("\n所有组合处理完成！")
    print("生成的可视化文件：")
    for result in results:
        print(f"- {result}")

if __name__ == "__main__":
    main() 