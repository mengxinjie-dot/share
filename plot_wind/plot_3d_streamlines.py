import pyvista as pv
import numpy as np
from nc_data_reader import NCDataReader
from pyvista import PolyData
from typing import Optional, Tuple, Dict

class WindStreamlinesPlotter:
    def __init__(self, 
                 file_path: str,
                 time_step: int = 100,
                 max_height: Optional[float] = None,
                 window_size: Tuple[int, int] = (1920, 1080),
                 colormap: str = 'rainbow',
                 terrain_colormap: str = 'terrain',
                 background_color: str = 'white',
                 source_points: Dict[str, int] = {'nx': 20, 'ny': 20, 'nz': 10},
                 streamline_config: Dict = None,
                 scalar_bar_config: Dict = None,
                 terrain_config: Dict = None):
        """
        风场流线图绘制器
        Args:
            file_path: nc文件路径
            time_step: 时间步长
            max_height: 最大绘制高度
            window_size: 窗口大小
            colormap: 流线颜色映射方案
            terrain_colormap: 地形颜色映射方案
            background_color: 背景颜色
            source_points: 流线起始点配置
            streamline_config: 流线配置
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
        self.source_points = source_points
        
        # 默认流线配置
        self.streamline_config = {
            'integration_direction': 'forward',
            'max_steps': 1000,
            'tube_radius_factor': 2000,  # 管道半径因子
            'opacity': 0.8,  # 流线不透明度
        }
        if streamline_config:
            self.streamline_config.update(streamline_config)
            
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
        
    def _load_data(self):
        """读取数据"""
        reader = NCDataReader(self.file_path)
        self.x, self.y, self.z, self.u, self.v, self.w = reader.read_wind_data(self.time_step)
        
        # 读取地形数据
        self.terrain_x, self.terrain_y, self.terrain_z = reader.get_terrain_surface(self.time_step)
        reader.close()
        
        # 如果设置了最大高度，裁剪数据
        if self.max_height is not None:
            z_mask = self.z <= self.max_height
            self.z = self.z[z_mask]
            self.u = self.u[z_mask]
            self.v = self.v[z_mask]
            self.w = self.w[z_mask]
            
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
    
    def _create_source_points(self):
        """创建流线起始点"""
        nx = self.source_points['nx']
        ny = self.source_points['ny']
        nz = self.source_points['nz']
        
        # x和y方向的起始点保持不变
        x_start = np.linspace(self.x.min(), self.x.max(), nx)
        y_start = np.linspace(self.y.min(), self.y.max(), ny)
        
        # 创建网格点
        xx, yy = np.meshgrid(x_start, y_start, indexing='ij')
        points = []
        
        # 对每个xy位置，根据地形高度创建z方向的起始点
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x_pos = xx[i, j]
                y_pos = yy[i, j]
                
                # 找到最接近的地形点
                x_idx = np.abs(self.terrain_x[:, 0] - x_pos).argmin()
                y_idx = np.abs(self.terrain_y[0, :] - y_pos).argmin()
                
                # 获取该位置的地形高度
                terrain_height = self.terrain_z[y_idx, x_idx]
                height_step = 50
                z_start = np.arange(terrain_height + height_step, terrain_height + height_step*nz, height_step)  # 从地形上方50米开始,每隔50米创建一个点
                
                # 添加所有起始点
                for z_pos in z_start:
                    points.append([x_pos, y_pos, z_pos])
        
        return PolyData(np.array(points))
    
    def plot(self, save_path: Optional[str] = None):
        """绘制流线图
        Args:
            save_path: 图片保存路径，如果为None则显示图形
        """
        # 创建3D网格
        grid = self._create_grid()
        
        # 创建plotter
        plotter = pv.Plotter(off_screen=save_path is not None, 
                            window_size=self.window_size)
        
        # 添加地形
        self._add_terrain(plotter)
        
        # 生成流线
        source = self._create_source_points()
        streamlines = grid.streamlines_from_source(
            source,
            vectors='vectors',
            integration_direction=self.streamline_config['integration_direction'],
            max_steps=self.streamline_config['max_steps'],
        )
        
        # 计算流线上的风速大小
        streamlines['velocity_magnitude'] = np.linalg.norm(streamlines['vectors'], axis=1)
        
        # 设置流线样式
        tube_radius = (max(self.x.max()-self.x.min(), 
                         self.y.max()-self.y.min(), 
                         self.z.max()-self.z.min()) 
                     / self.streamline_config['tube_radius_factor'])
        
        plotter.add_mesh(streamlines.tube(radius=tube_radius), 
                        scalars='velocity_magnitude',
                        cmap=self.colormap,
                        show_scalar_bar=True,
                        smooth_shading=True,
                        opacity=self.streamline_config['opacity'],
                        scalar_bar_args=self.scalar_bar_config)
        
        # 设置视角和背景
        plotter.set_background(self.background_color)
        plotter.camera_position = 'xz'
        plotter.camera.zoom(1.5)
        plotter.camera.elevation = 30
        plotter.camera.azimuth = 0
        
        # 添加指北和指东箭头
        # 计算箭头起点位置（在视图右上角）
        x_range = self.x.max() - self.x.min()
        y_range = self.y.max() - self.y.min()
        z_range = self.z.max() - self.z.min()
        
        # 修改箭头起点位置到左上角
        arrow_start = np.array([
            self.x.min() + x_range * 0.1,  # x坐标，靠近左边
            self.y.min() + y_range * 0.1,  # y坐标，靠近前边
            self.z.max() - z_range * 0.1   # z坐标，靠近顶部
        ])
        
        # 箭头长度为范围的10%
        arrow_length = min(x_range, y_range) * 0.1
        
        # 添加指北箭头（红色）- y轴正方向
        north_end = arrow_start + np.array([0, arrow_length, 0])
        plotter.add_arrows(
            arrow_start[None, :],
            north_end[None, :] - arrow_start[None, :],
            color='red',
        )
        plotter.add_text(
            'N',
            position=(north_end[0], north_end[1], north_end[2]),
            font_size=14,
            color='red'
        )
        
        # 添加指东箭头（蓝色）- x轴正方向
        east_end = arrow_start + np.array([arrow_length, 0, 0])
        plotter.add_arrows(
            arrow_start[None, :],
            east_end[None, :] - arrow_start[None, :],
            color='blue',
        )
        plotter.add_text(
            'E',
            position=(east_end[0], east_end[1], east_end[2]),
            font_size=14,
            color='blue'
        )
        
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
        plotter.add_title(f"3D Wind Streamlines Visualization{height_info}", 
                         font_size=20, 
                         color='black')
        
        # 保存或显示图形
        if save_path is not None:
            plotter.screenshot(save_path, return_img=False, 
                             transparent_background=False)
            plotter.close()
        else:
            plotter.show()

if __name__ == '__main__':
    # 示例配置
    streamline_config = {
        'integration_direction': 'forward',
        'max_steps': 1000,
        'tube_radius_factor': 2000,
        'opacity': 1,
    }
    
    scalar_bar_config = {
        'title': 'Wind Speed (m/s)',
        'position_x': 0.85,
        'position_y': 0.05,
        'width': 0.1,
        'height': 0.1,
    }
    
    source_points = {
        'nx': 60,
        'ny': 60,
        'nz': 4,
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
    
    # 创建绘图器实例
    plotter = WindStreamlinesPlotter(
        file_path='/home/mxj/docker_palm/share/tongren02_OUTPUT/tongren02_3d.000.nc',
        time_step=150,
        max_height=2000,
        colormap='rainbow',
        terrain_colormap='terrain',  # 使用terrain配色方案表示地形高度
        source_points=source_points,
        streamline_config=streamline_config,
        scalar_bar_config=scalar_bar_config,
        terrain_config=terrain_config
    )
    
    # 绘制并保存图片
    plotter.plot(save_path='/home/mxj/docker_palm/share/picture/wind_streamlines_3d.png') 