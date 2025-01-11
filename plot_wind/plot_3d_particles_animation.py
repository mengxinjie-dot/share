import pyvista as pv
import numpy as np
from nc_data_reader import NCDataReader
from pyvista import PolyData
from typing import Optional, Tuple, Dict
import time
import imageio.v3 as iio
import os
import vtk

class WindParticlesAnimator:
    def __init__(self, 
                 file_path: str,
                 time_step: int = 100,
                 max_height: Optional[float] = None,
                 window_size: Tuple[int, int] = (1920, 1080),
                 colormap: str = 'rainbow',
                 terrain_colormap: str = 'terrain',
                 background_color: str = 'white',
                 particle_config: Dict = None,
                 scalar_bar_config: Dict = None,
                 terrain_config: Dict = None):
        """
        风场粒子动画器
        Args:
            file_path: nc文件路径
            time_step: 时间步长
            max_height: 最大绘制高度
            window_size: 窗口大小
            colormap: 粒子颜色映射方案
            terrain_colormap: 地形颜色映射方案
            background_color: 背景颜色
            particle_config: 粒子配置
            scalar_bar_config: 标量条配置
            terrain_config: 地形显示配置
        """
        self.file_path = file_path
        self.time_step = time_step
        self.max_height = max_height
        self.window_size = window_size
        self.colormap = colormap
        self.terrain_colormap = terrain_colormap
        self.background_color = background_color
        
        # 默认粒子配置
        self.particle_config = {
            'n_particles': 5000,  # 粒子数量
            'particle_size': 5.0,  # 粒子大小
            'particle_opacity': 0.8,  # 粒子不透明度
            'max_time': 10.0,  # 动画最大时长(秒)
            'dt': 0.1,  # 时间步长(秒)
            'speed_scale': 1.0,  # 速度缩放因子
            'trail_length': 10,  # 尾迹长度（保存的历史位置数量）
            'trail_decay': 0.8,  # 尾迹衰减系数（每个历史点的不透明度衰减）
        }
        if particle_config:
            self.particle_config.update(particle_config)
            
        # 默认标量条配置
        self.scalar_bar_config = {
            'title': 'Wind Speed (m/s)',
            'title_font_size': 16,
            'label_font_size': 14,
            'position_x': 0.85,
            'position_y': 0.05,
            'width': 0.1,
            'height': 0.1,
            'fmt': '%.1f',
            'color': 'black',
        }
        if scalar_bar_config:
            self.scalar_bar_config.update(scalar_bar_config)
            
        # 默认地形配置
        self.terrain_config = {
            'opacity': 1.0,
            'specular': 0.5,
            'specular_power': 15,
            'ambient': 0.3,
            'diffuse': 0.8,
            'smooth_shading': True,
            'show_edges': False,
            'show_scalar_bar': True,
            'scalar_bar_args': {
                'title': 'Terrain Height (m)',
                'title_font_size': 16,
                'label_font_size': 14,
                'position_x': 0.05,
                'position_y': 0.05,
                'width': 0.1,
                'height': 0.1,
                'fmt': '%.0f',
                'color': 'black',
            }
        }
        if terrain_config:
            self.terrain_config.update(terrain_config)
            
        # 读取数据
        self._load_data()
        
        # 初始化粒子历史位置列表
        self.particle_history = []
        
    def _load_data(self):
        """读取数据"""
        reader = NCDataReader(self.file_path)
        self.x, self.y, self.z, u, v, w = reader.read_wind_data(self.time_step)
        
        # 读取地形数据
        self.terrain_x, self.terrain_y, self.terrain_z = reader.get_terrain_surface(self.time_step)
        reader.close()
        
        # 调整数据维度顺序从(zu_3d, y, xu)到(y, xu, zu_3d)
        self.u = np.transpose(u, (1, 2, 0))  # (zu_3d, y, xu) -> (y, xu, zu_3d)
        self.v = np.transpose(v, (1, 2, 0))  # (zu_3d, yv, x) -> (yv, x, zu_3d)
        self.w = np.transpose(w, (1, 2, 0))  # (zw_3d, y, x) -> (y, x, zw_3d)
        
        # 如果设置了最大高度，裁剪数据
        if self.max_height is not None:
            z_mask = self.z <= self.max_height
            self.z = self.z[z_mask]
            self.u = self.u[:, :, z_mask]
            self.v = self.v[:, :, z_mask]
            self.w = self.w[:, :, z_mask]
            
    def _add_terrain(self, plotter):
        """添加地形到场景中"""
        if self.terrain_x is not None:
            # 创建地形网格
            terrain_grid = pv.StructuredGrid(self.terrain_x, self.terrain_y, self.terrain_z)
            
            # 添加高度数据作为标量值
            terrain_grid['height'] = self.terrain_z.ravel()
            
            # 设置地形配置
            terrain_args = self.terrain_config.copy()
            scalar_bar_args = terrain_args.pop('scalar_bar_args')
            show_scalar_bar = terrain_args.pop('show_scalar_bar')
            
            # 添加地形到场景
            plotter.add_mesh(terrain_grid,
                           scalars='height',
                           cmap=self.terrain_colormap,
                           show_scalar_bar=show_scalar_bar,
                           scalar_bar_args=scalar_bar_args,
                           **terrain_args)
    
    def _create_grid(self):
        """创建3D网格"""
        x_mesh, y_mesh, z_mesh = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        grid = pv.StructuredGrid(x_mesh, y_mesh, z_mesh)
        
        # 计算风速大小并添加数据
        velocity_magnitude = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        vectors = np.stack((self.u.ravel(), self.v.ravel(), self.w.ravel()), axis=1)
        grid['vectors'] = vectors
        grid['velocity_magnitude'] = velocity_magnitude.ravel()
        
        return grid
    
    def _initialize_particles(self):
        """初始化粒子位置"""
        n_particles = self.particle_config['n_particles']
        
        # 在空间中随机生成粒子，但避免边界
        margin = 50  # 边界缓冲区（米）
        x_particles = np.random.uniform(self.x.min() + margin, self.x.max() - margin, n_particles)
        y_particles = np.random.uniform(self.y.min() + margin, self.y.max() - margin, n_particles)
        
        # 确保粒子初始位置在地形之上
        z_particles = []
        for i in range(n_particles):
            # 找到最接近的地形点
            x_idx = np.abs(self.terrain_x[0, :] - x_particles[i]).argmin()
            y_idx = np.abs(self.terrain_y[:, 0] - y_particles[i]).argmin()
            
            # 获取该位置的地形高度
            terrain_height = self.terrain_z[y_idx, x_idx]
            
            # 在地形上方随机生成高度，添加最小高度限制
            min_height = terrain_height + 50  # 确保粒子在地形上方至少50米
            if self.max_height is not None:
                max_z = min(self.max_height, self.z.max())
            else:
                max_z = self.z.max()
            
            z_particles.append(np.random.uniform(min_height, max_z))
        
        z_particles = np.array(z_particles)
        
        return np.column_stack((x_particles, y_particles, z_particles)), None
    
    def _interpolate_velocity(self, points):
        """在给定位置插值计算速度"""
        from scipy.interpolate import RegularGridInterpolator
        
        try:
            # 确保点在有效范围内
            x_valid = np.clip(points[:, 0], self.x.min(), self.x.max())
            y_valid = np.clip(points[:, 1], self.y.min(), self.y.max())
            z_valid = np.clip(points[:, 2], self.z.min(), self.z.max())
            points_valid = np.column_stack((x_valid, y_valid, z_valid))
            
            # 创建插值器，注意坐标轴的顺序
            u_interp = RegularGridInterpolator((self.y, self.x, self.z), self.u, bounds_error=False, fill_value=0)
            v_interp = RegularGridInterpolator((self.y, self.x, self.z), self.v, bounds_error=False, fill_value=0)
            w_interp = RegularGridInterpolator((self.y, self.x, self.z), self.w, bounds_error=False, fill_value=0)
            
            # 计算速度
            u = u_interp(points_valid[:, [1, 0, 2]])  # 注意坐标轴顺序：y, x, z
            v = v_interp(points_valid[:, [1, 0, 2]])
            w = w_interp(points_valid[:, [1, 0, 2]])
            
            return np.column_stack((u, v, w))
        except Exception as e:
            print(f"插值错误: {str(e)}")
            return np.zeros_like(points)
    
    def _update_particles(self, points, dt):
        """更新粒子位置"""
        # 获取当前位置的速度
        velocities = self._interpolate_velocity(points)
        
        # 更新位置，增加速度缩放以使运动更明显
        new_points = points + velocities * dt * self.particle_config['speed_scale']
        
        # 处理超出边界的粒子
        x_out = (new_points[:, 0] < self.x.min()) | (new_points[:, 0] > self.x.max())
        y_out = (new_points[:, 1] < self.y.min()) | (new_points[:, 1] > self.y.max())
        z_out = (new_points[:, 2] < self.z.min()) | (new_points[:, 2] > self.z.max())
        
        # 检查是否碰到地形
        terrain_collision = np.zeros(len(points), dtype=bool)
        for i in range(len(points)):
            try:
                x_idx = np.abs(self.terrain_x[0, :] - new_points[i, 0]).argmin()
                y_idx = np.abs(self.terrain_y[:, 0] - new_points[i, 1]).argmin()
                terrain_height = self.terrain_z[y_idx, x_idx]
                terrain_collision[i] = new_points[i, 2] < terrain_height + 5  # 添加一个小的缓冲区
            except IndexError as e:
                print(f"地形碰撞检测错误: {str(e)}")
                terrain_collision[i] = True
        
        out_of_bounds = x_out | y_out | z_out | terrain_collision
        
        # 重新初始化超出边界的粒子
        if np.any(out_of_bounds):
            new_particles, _ = self._initialize_particles()
            new_points[out_of_bounds] = new_particles[out_of_bounds]
        
        return new_points
    
    def animate(self, save_path: Optional[str] = None):
        """创建粒子动画"""
        # 创建plotter
        plotter = pv.Plotter(off_screen=save_path is not None, 
                            window_size=self.window_size)
        
        # 添加地形
        self._add_terrain(plotter)
        
        # 初始化粒子
        points, _ = self._initialize_particles()
        particles = PolyData(points)
        
        # 初始化粒子历史位置
        self.particle_history = [points]
        
        # 计算初始速度和颜色
        velocities = self._interpolate_velocity(points)
        speeds = np.linalg.norm(velocities, axis=1)
        particles['speed'] = speeds
        
        # 添加粒子到场景
        particles_actor = plotter.add_mesh(particles,
                                         scalars='speed',
                                         render_points_as_spheres=True,
                                         point_size=self.particle_config['particle_size'],
                                         opacity=self.particle_config['particle_opacity'],
                                         cmap=self.colormap,
                                         show_scalar_bar=True,
                                         scalar_bar_args=self.scalar_bar_config)
        
        # 创建尾迹actors列表
        trail_actors = []
        
        # 设置视角和背景
        plotter.set_background(self.background_color)
        plotter.camera_position = 'xz'
        plotter.camera.zoom(1.5)
        plotter.camera.elevation = 30
        plotter.camera.azimuth = 0
        
        # 添加网格坐标轴
        plotter.show_grid(
            xtitle='X (m)',
            ytitle='Y (m)',
            ztitle='Z (m)',
            grid=True,
            color='black',
            font_size=14,
            show_zlabels=True,
        )
        
        # 添加标题
        height_info = f" (Height ≤ {self.max_height}m)" if self.max_height is not None else ""
        plotter.add_title(f"3D Wind Particles Animation{height_info}", 
                         font_size=20, 
                         color='black')
        
        # 动画循环
        plotter.show(auto_close=False)
        start_time = time.time()
        
        try:
            # 如果需要保存动画，准备帧列表
            frames = []
            
            while (time.time() - start_time) < self.particle_config['max_time']:
                # 更新粒子位置
                points = self._update_particles(points, self.particle_config['dt'])
                
                # 更新粒子历史位置
                self.particle_history.append(points)
                if len(self.particle_history) > self.particle_config['trail_length']:
                    self.particle_history.pop(0)
                
                # 移除旧的尾迹actors
                for actor in trail_actors:
                    plotter.remove_actor(actor)
                trail_actors.clear()
                
                # 绘制尾迹
                for i, hist_points in enumerate(self.particle_history[:-1]):
                    # 计算当前历史点的不透明度
                    opacity = self.particle_config['particle_opacity'] * (
                        self.particle_config['trail_decay'] ** (len(self.particle_history) - i - 1)
                    )
                    
                    # 创建历史点的PolyData
                    hist_particles = PolyData(hist_points)
                    
                    # 计算历史点的速度和颜色
                    hist_velocities = self._interpolate_velocity(hist_points)
                    hist_speeds = np.linalg.norm(hist_velocities, axis=1)
                    hist_particles['speed'] = hist_speeds
                    
                    # 添加历史点到场景
                    trail_actor = plotter.add_mesh(hist_particles,
                                                 scalars='speed',
                                                 render_points_as_spheres=True,
                                                 point_size=self.particle_config['particle_size'] * 0.8,  # 尾迹点略小
                                                 opacity=opacity,
                                                 cmap=self.colormap,
                                                 show_scalar_bar=False)
                    trail_actors.append(trail_actor)
                
                # 更新当前粒子
                velocities = self._interpolate_velocity(points)
                speeds = np.linalg.norm(velocities, axis=1)
                particles.points = points
                particles['speed'] = speeds
                
                plotter.render()
                
                if save_path is not None:
                    # 获取当前帧图像
                    frame = plotter.screenshot(None, return_img=True)
                    frames.append(frame)
                
                time.sleep(self.particle_config['dt'])
            
            # 如果需要保存动画，将帧保存为GIF
            if save_path is not None and frames:
                # 修改文件扩展名为.gif
                gif_path = os.path.splitext(save_path)[0] + '.gif'
                
                # 计算帧率
                fps = int(1.0 / self.particle_config['dt'])
                duration = 1000 / fps  # 转换为毫秒
                
                # 保存为GIF
                iio.imwrite(
                    gif_path,
                    frames,
                    duration=duration,  # 每帧持续时间（毫秒）
                )
                
                print(f"动画已保存到: {gif_path}")
                
        except Exception as e:
            print(f"动画生成过程中发生错误: {str(e)}")
        finally:
            # 关闭plotter
            plotter.close()

if __name__ == '__main__':
    # 示例配置
    particle_config = {
        'n_particles': 1000,      # 减少粒子数量以提高性能
        'particle_size': 3.0,     # 较小的粒子
        'particle_opacity': 1,
        'max_time': 1000.0,
        'dt': 0.05,
        'speed_scale': 1000.0,    # 保持较大的速度缩放因子
        'trail_length': 15,      # 尾迹长度
        'trail_decay': 0.8,      # 尾迹衰减系数
    }
    
    scalar_bar_config = {
        'title': 'Wind Speed (m/s)',
        'position_x': 0.85,
        'position_y': 0.05,
        'width': 0.1,
        'height': 0.1,
    }
    
    terrain_config = {
        'opacity': 1.0,
        'specular': 0.5,
        'ambient': 0.3,
        'show_edges': False,
        'show_scalar_bar': True,
        'scalar_bar_args': {
            'title': 'Terrain Height (m)',
            'position_x': 0.05,
            'position_y': 0.05,
            'width': 0.1,
            'height': 0.1,
        }
    }
    
    # 创建动画器实例
    animator = WindParticlesAnimator(
        file_path='/home/mxj/docker_palm/share/tongren02_OUTPUT/tongren02_3d.000.nc',
        time_step=150,
        max_height=2000,
        window_size=(1280, 720),  # 减小窗口大小以提高性能
        colormap='Reds',
        terrain_colormap='terrain',
        particle_config=particle_config,
        scalar_bar_config=scalar_bar_config,
        terrain_config=terrain_config
    )
    
    # 创建并保存动画
    animator.animate(save_path='/home/mxj/docker_palm/share/animations/wind_particles_animation.gif') 