import pyvista as pv
import numpy as np
from nc_data_reader import NCDataReader
from plot_3d_wind_visualization import WindVisualizationPlotter
from typing import Optional, Tuple, Dict

class WindVolumeRenderer(WindVisualizationPlotter):
    def __init__(self, 
                 file_path: str,
                 time_step: int = 100,
                 max_height: Optional[float] = None,
                 window_size: Tuple[int, int] = (1920, 1080),
                 colormap: str = 'rainbow',
                 terrain_colormap: str = 'terrain',
                 background_color: str = 'white',
                 volume_config: Dict = None,
                 scalar_bar_config: Dict = None,
                 terrain_config: Dict = None):
        """
        风场体渲染器
        Args:
            file_path: nc文件路径
            time_step: 时间步长
            max_height: 最大绘制高度
            window_size: 窗口大小
            colormap: 体渲染颜色映射方案
            terrain_colormap: 地形颜色映射方案
            background_color: 背景颜色
            volume_config: 体渲染配置
            scalar_bar_config: 标量条配置
            terrain_config: 地形显示配置
        """
        # 默认体渲染配置
        self.volume_config = {
            'opacity': [0, 0.2, 0.4, 0.6, 0.8],  # 不同值的不透明度
            'opacity_unit_distance': 100,  # 不透明度单位距离
            'mapper': 'smart',  # 映射方式
            'blend_mode': 'composite',  # 混合模式
            'ambient': 0.2,  # 环境光
            'diffuse': 0.8,  # 漫反射
            'specular': 0.3,  # 镜面反射
            'shade': True,  # 是否启用阴影
        }
        if volume_config:
            self.volume_config.update(volume_config)
            
        super().__init__(
            file_path=file_path,
            time_step=time_step,
            max_height=max_height,
            window_size=window_size,
            colormap=colormap,
            terrain_colormap=terrain_colormap,
            background_color=background_color,
            scalar_bar_config=scalar_bar_config,
            terrain_config=terrain_config
        )
    
    def _add_volume(self, plotter, grid):
        """添加体渲染"""
        # 将StructuredGrid转换为ImageData
        # 计算网格间距
        spacing = (
            (self.x.max() - self.x.min()) / (len(self.x) - 1),
            (self.y.max() - self.y.min()) / (len(self.y) - 1),
            (self.z.max() - self.z.min()) / (len(self.z) - 1)
        )
        
        # 创建ImageData
        volume = pv.ImageData(
            dimensions=(len(self.x), len(self.y), len(self.z)),
            spacing=spacing,
            origin=(self.x.min(), self.y.min(), self.z.min())
        )
        
        # 将数据从StructuredGrid插值到ImageData
        volume.point_data['velocity_magnitude'] = grid.point_data['velocity_magnitude']
        
        # 添加体渲染
        plotter.add_volume(
            volume,
            scalars='velocity_magnitude',
            cmap='Reds',
            opacity='sigmoid',
            mapper=self.volume_config['mapper'],
            blending=self.volume_config['blend_mode'],
            ambient=self.volume_config['ambient'],
            diffuse=self.volume_config['diffuse'],
            specular=self.volume_config['specular'],
            shade=self.volume_config['shade'],
            opacity_unit_distance=self.volume_config['opacity_unit_distance'],
            show_scalar_bar=False,
        )
        
        # 添加标量条
        plotter.add_scalar_bar(**self.scalar_bar_config)
        
        return volume
    
    def plot(self, save_path: Optional[str] = None):
        """绘制体渲染图
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
        
        # 添加箭头场
        # self._add_arrows(plotter, grid)
        
        # 添加体渲染
        self._add_volume(plotter, grid)
        
        # 设置视角和背景
        plotter.set_background(self.background_color)
        plotter.camera_position = 'xz'
        plotter.camera.zoom(1.5)
        plotter.camera.elevation = 30
        plotter.camera.azimuth = 0
        
        # 添加指北和指东箭头
        x_range = self.x.max() - self.x.min()
        y_range = self.y.max() - self.y.min()
        z_range = self.z.max() - self.z.min()
        
        arrow_start = np.array([
            self.x.min() + x_range * 0.1,
            self.y.min() + y_range * 0.1,
            self.z.max() - z_range * 0.1
        ])
        
        arrow_length = min(x_range, y_range) * 0.1
        
        # 添加指北箭头（红色）
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
        
        # 添加指东箭头（蓝色）
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
        plotter.add_title(f"3D Wind Volume Rendering{height_info}", 
                         font_size=20, 
                         color='black')
        
        # 保存或显示图形
        if save_path is not None:
            plotter.screenshot(save_path, return_img=False, 
                               window_size=self.window_size,
                             transparent_background=False)
            plotter.close()
        else:
            plotter.show()

if __name__ == '__main__':
    # 示例配置
    volume_config = {
        'opacity': [0, 0.2, 0.4, 0.6, 0.8],  # 修改默认不透明度值
        'opacity_unit_distance': 500,
        'mapper': 'smart',
        'blend_mode': 'composite',
        'ambient': 0.2,
        'diffuse': 0.8,
        'specular': 0.3,
        'shade': False,
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
    
    # 创建体渲染器实例
    renderer = WindVolumeRenderer(
        file_path='/home/mxj/docker_palm/share/tongren02_OUTPUT/tongren02_3d.000.nc',
        time_step=150,
        max_height=2000,
        colormap='rainbow',
        terrain_colormap='terrain',
        volume_config=volume_config,
        scalar_bar_config=scalar_bar_config,
        terrain_config=terrain_config
    )
    
    # 绘制并保存体渲染图
    renderer.plot(save_path='/home/mxj/docker_palm/share/picture/wind_volume_3d.png') 