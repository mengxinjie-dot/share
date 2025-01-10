import pyvista as pv
import numpy as np
from plot_3d_streamlines import WindStreamlinesPlotter
from typing import Optional, Tuple, Dict, Literal

class WindVisualizationPlotter(WindStreamlinesPlotter):
    def __init__(self, 
                 file_path: str,
                 visualization_type: Literal['streamlines', 'arrows'] = 'streamlines',
                 time_step: int = 100,
                 max_height: Optional[float] = None,
                 window_size: Tuple[int, int] = (1920, 1080),
                 colormap: str = 'rainbow',
                 terrain_colormap: str = 'terrain',
                 background_color: str = 'white',
                 source_points: Dict[str, int] = {'nx': 20, 'ny': 20, 'nz': 10},
                 streamline_config: Dict = None,
                 arrow_config: Dict = None,
                 scalar_bar_config: Dict = None,
                 terrain_config: Dict = None):
        """
        风场可视化绘制器，支持流线图和箭头场图
        Args:
            file_path: nc文件路径
            visualization_type: 可视化类型，'streamlines' 为流线图，'arrows' 为箭头场图
            time_step: 时间步长
            max_height: 最大绘制高度
            window_size: 窗口大小
            colormap: 颜色映射方案
            terrain_colormap: 地形颜色映射方案
            background_color: 背景颜色
            source_points: 流线/箭头起始点配置
            streamline_config: 流线配置
            arrow_config: 箭头配置
            scalar_bar_config: 标量条配置
            terrain_config: 地形显示配置
        """
        self.visualization_type = visualization_type
        
        # 默认箭头配置
        self.arrow_config = {
            'scale': 200,  # 箭头缩放因子
            'shaft_radius': 0.15,  # 箭头轴半径
            'tip_radius': 0.3,  # 箭头头部半径
            'tip_length': 0.5,  # 箭头头部长度
            'opacity': 0.8,  # 箭头不透明度
        }
        if arrow_config:
            self.arrow_config.update(arrow_config)
            
        super().__init__(
            file_path=file_path,
            time_step=time_step,
            max_height=max_height,
            window_size=window_size,
            colormap=colormap,
            terrain_colormap=terrain_colormap,
            background_color=background_color,
            source_points=source_points,
            streamline_config=streamline_config,
            scalar_bar_config=scalar_bar_config,
            terrain_config=terrain_config
        )
    
    def _add_arrows(self, plotter, grid):
        """添加箭头场"""
        # 创建箭头起始点
        source = self._create_source_points()
        points = source.points
        
        # 在每个起始点计算风速向量
        vectors = []
        magnitudes = []
        for point in points:
            # 找到最近的网格点，同时确保索引在有效范围内
            i = min(max(0, np.abs(self.z - point[2]).argmin()), self.z.size - 1)  # z方向
            j = min(max(0, np.abs(self.y - point[1]).argmin()), self.y.size - 1)  # y方向
            k = min(max(0, np.abs(self.x - point[0]).argmin()), self.x.size - 1)  # x方向
            
            # 获取该点的风速分量
            u_val = self.u[i, j, k]
            v_val = self.v[i, j, k]
            w_val = self.w[i, j, k]
            
            vector = np.array([u_val, v_val, w_val])
            vectors.append(vector)
            magnitudes.append(np.linalg.norm(vector))
        
        vectors = np.array(vectors)
        magnitudes = np.array(magnitudes)
        
        # 归一化向量并应用缩放
        scale = self.arrow_config['scale']
        normalized_vectors = vectors / np.max(magnitudes) * scale
        
        # 创建箭头几何体
        arrows = pv.Arrow()  # 创建单个箭头模板
        arrow_collection = pv.PolyData(points)  # 创建箭头集合
        
        # 添加向量和大小数据
        arrow_collection.point_data['vectors'] = normalized_vectors
        arrow_collection.point_data['magnitude'] = magnitudes
        
        # 使用 glyph 创建箭头场
        arrows = arrow_collection.glyph(
            orient='vectors',
            scale='magnitude',
            factor=scale * 0.5,  # 缩放因子
            geom=arrows,
        )
        
        # 添加箭头到场景
        plotter.add_mesh(
            arrows,
            scalars='magnitude',
            cmap=self.colormap,
            opacity=self.arrow_config['opacity'],
            show_scalar_bar=False  # 禁用自动添加的标量条
        )
        
        # 添加标量条
        plotter.add_scalar_bar(**self.scalar_bar_config)
        
        return arrows
    
    def plot(self, save_path: Optional[str] = None):
        """绘制风场图
        Args:
            save_path: 图片保存路径，如果为None则显示图形
        """
        if self.visualization_type == 'streamlines':
            # 使用父类的绘图方法绘制流线图
            super().plot(save_path)
        else:
            # 创建3D网格
            grid = self._create_grid()
            
            # 创建plotter
            plotter = pv.Plotter(off_screen=save_path is not None, 
                                window_size=self.window_size)
            
            # 添加地形
            self._add_terrain(plotter)
            
            # 添加箭头场
            self._add_arrows(plotter, grid)
            
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
            plotter.add_title(f"3D Wind Arrows Visualization{height_info}", 
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
    
    arrow_config = {
        'scale': 100,
        'shaft_radius': 0.15,
        'tip_radius': 0.3,
        'tip_length': 0.5,
        'opacity': 0.8,
    }
    
    scalar_bar_config = {
        'title': 'Wind Speed (m/s)',
        'position_x': 0.85,
        'position_y': 0.05,
        'width': 0.1,
        'height': 0.1,
    }
    
    source_points = {
        'nx': 80,
        'ny': 80,
        'nz': 6,
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
    
    # 创建绘图器实例 - 箭头场图
    plotter_arrows = WindVisualizationPlotter(
        file_path='/home/mxj/docker_palm/share/tongren02_OUTPUT/tongren02_3d.000.nc',
        visualization_type='arrows',
        time_step=150,
        max_height=2000,
        colormap='rainbow',
        terrain_colormap='terrain',
        source_points=source_points,
        arrow_config=arrow_config,
        scalar_bar_config=scalar_bar_config,
        terrain_config=terrain_config
    )
    
    # 绘制并保存箭头场图
    plotter_arrows.plot(save_path='/home/mxj/docker_palm/share/picture/wind_arrows_3d.png') 