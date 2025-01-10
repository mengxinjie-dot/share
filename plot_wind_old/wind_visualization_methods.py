import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import pyvista as pv

class WindVisualizer:
    def __init__(self, mode,x, y, z, u, v, w=None, wind_speed=None):
        """
        初始化风场可视化器
        x, y, z: 3D网格坐标
        u, v, w: 风场分量（u:x方向，v:y方向，w:z方向）
        """
        self.mode = mode
        self.x = x
        self.y = y
        self.z = z
        self.u = u
        self.v = v
        print("u.shape,v.shape,w.shape",u.shape,v.shape,w.shape)
        self.w = w if w is not None else np.zeros_like(u)
        self.wind_speed = wind_speed

    def setup_3d_axes(self, figsize=(12, 8)):
        """设置3D图形基本属性"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        return fig, ax

    def add_direction_arrows(self, ax):
        """添加方向指示箭头"""
        x_range = np.max(self.x) - np.min(self.x)
        y_range = np.max(self.y) - np.min(self.y)
        z_range = np.max(self.z) - np.min(self.z)
        
        arrow_length = x_range * 0.1
        arrow_x = np.max(self.x) - arrow_length * 0.5
        arrow_y = np.max(self.y) - arrow_length * 0.5
        arrow_z = np.max(self.z) * 1.2
        
        # 添加指北箭头（红色）
        ax.quiver(arrow_x, arrow_y, arrow_z, 0, arrow_length, 0,
                 color='red', arrow_length_ratio=0.2, linewidth=2)
        # 添加指东箭头（蓝色）
        ax.quiver(arrow_x, arrow_y, arrow_z, arrow_length, 0, 0,
                 color='blue', arrow_length_ratio=0.2, linewidth=2)
        
        ax.text(arrow_x + arrow_length * 1.2, arrow_y, arrow_z, 'E', 
                color='blue', fontsize=12, fontweight='bold')
        ax.text(arrow_x, arrow_y + arrow_length * 1.2, arrow_z, 'N', 
                color='red', fontsize=12, fontweight='bold')

    def scatter_plot(self, output_path, n_layers=None, display_topo=False):
        """3D散点图可视化"""
        fig, ax = self.setup_3d_axes()
        
        xx, yy = np.meshgrid(self.x, self.y, indexing='ij')
        points = []
        colors_array = []
        
        if self.mode == 'bottom':
            z_range = n_layers
        else:
            z_range = len(self.z)
            
        for k in range(z_range):
            mask = ~np.ma.getmask(self.wind_speed[:, :, k])
            points.extend(zip(xx[mask].flatten(), 
                            yy[mask].flatten(), 
                            np.full_like(xx[mask].flatten(), self.z[k])))
            colors_array.extend(self.wind_speed[:, :, k][mask].flatten())
            
            # 如果需要显示地形,将地形mask部分添加为深褐色点
            if display_topo:
                # 获取当前层的mask
                topo_mask = np.ma.getmask(self.wind_speed[:, :, k])
                
                # 获取每个xy点的最高有效层索引
                max_valid_z = np.zeros_like(xx, dtype=int)
                for i in range(xx.shape[0]):
                    for j in range(xx.shape[1]):
                        valid_indices = np.where(~np.ma.getmask(self.wind_speed[i,j,:]))[0]
                        if len(valid_indices) > 0:
                            max_valid_z[i,j] = np.max(valid_indices)
                
                # 只显示每个xy点有效值以下的地形点
                for i in range(xx.shape[0]):
                    for j in range(xx.shape[1]):
                        if topo_mask[i,j] and k <= max_valid_z[i,j]:
                            points.append((xx[i,j], yy[i,j], self.z[k]))
                            colors_array.append(-1)  # 用-1标记地形点
            
        points = np.array(points)
        colors_array = np.array(colors_array)
        
        # 创建自定义colormap
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        
        # 定义深红褐色
        dark_red = '#680034'
        
        # 创建包含深褐色的自定义colormap
        colors_under = plt.cm.jet(np.linspace(0, 1, 256))
        custom_cmap = ListedColormap(colors_under)
        custom_cmap.set_under(dark_red)  # 设置-1值对应的颜色
        
        # 计算坐标范围
        x_range = np.max(self.x) - np.min(self.x)
        y_range = np.max(self.y) - np.min(self.y)

        # 根据模式选择合适的z范围计算方法
        valid_z_indices = np.unique(np.where(~np.ma.getmask(self.wind_speed))[2])
        if len(valid_z_indices) > 0:
            max_z_index = np.max(valid_z_indices)
            min_z_index = np.min(valid_z_indices)
            z_range = self.z[max_z_index] - self.z[min_z_index]
        else:
            z_range = np.max(self.z[:n_layers]) - np.min(self.z[:n_layers])

        # 修改scatter的参数，使用自定义colormap
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=colors_array, cmap=custom_cmap, 
                            vmin=0,  # 设置一个接近-1的值作为最小值
                            s=1, alpha=0.1)
        
        self.add_direction_arrows(ax)
        ax.view_init(elev=30, azim=290)
        # 设置坐标轴比例相等
        ax.set_box_aspect([x_range/x_range, y_range/x_range, z_range/x_range])
        
        # 调整图形布局，确保所有标签都可见
        plt.tight_layout(pad=2.0)  # 增加padding

        # 设置z轴刻度和标签
        z_ticks = ax.get_zticks()
        ax.set_zticks(z_ticks[::2])  # 减少刻度数量

        plt.colorbar(scatter, label='Wind Speed (m/s)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.title('3D Wind Field - Scatter Plot')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def streamline_plot(self, output_path, n_layers, density=1):
        """使用PyVista实现3D流线图可视化"""
        # 创建结构化网格
        grid = pv.StructuredGrid()
        
        # 准备网格点坐标
        xx, yy, zz = np.meshgrid(self.x, self.y, self.z[:n_layers], indexing='ij')
        grid.points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
        grid.dimensions = [len(self.x), len(self.y), n_layers]
        
        # 准备速度场数据
        u_data = self.u[:, :, :n_layers]
        v_data = self.v[:, :, :n_layers]
        w_data = self.w[:, :, :n_layers] if self.w is not None else np.zeros_like(u_data)
        
        # 添加速度场数据到网格
        vectors = np.column_stack((
            u_data.flatten(),
            v_data.flatten(),
            w_data.flatten()
        ))
        grid['vectors'] = vectors
        
        # 添加风速大小数据
        if self.wind_speed is not None:
            grid['wind_speed'] = self.wind_speed[:, :, :n_layers].flatten()
        
        # 创建PyVista绘图器
        plotter = pv.Plotter(off_screen=True)
        
        # 设置背景颜色
        plotter.set_background('white')
        
        # 计算种子点
        n_seeds = int(min(grid.dimensions) * density)
        seeds = grid.points[::n_seeds]
        
        # 生成流线
        streamlines = grid.streamlines(
            vectors='vectors',
            source_radius=grid.length/10,
            n_points=n_seeds,
            source_center=grid.center,
            integration_direction='both',
            max_steps=1000,
        )
        
        # 根据风速设置流线颜色
        if self.wind_speed is not None:
            streamlines = streamlines.tube(radius=grid.length/500)
            plotter.add_mesh(streamlines, scalars='vectors_mag', 
                            cmap='jet', 
                            line_width=1,
                            show_scalar_bar=True,
                            scalar_bar_args={'title': 'Wind Speed (m/s)'})
        
        # 设置相机位置
        plotter.camera_position = 'xz'
        plotter.camera.elevation = 30
        plotter.camera.azimuth = 290
        
        # 添加坐标轴标签
        plotter.add_axes(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        
        # 保存图像
        plotter.show(screenshot=output_path)
        plotter.close()

    def slice_plot(self, output_path, z_levels=3):
        """切片图可视化"""
        fig, ax = self.setup_3d_axes()
        
        # 选择要显示的z层
        z_indices = np.linspace(0, len(self.z)-1, z_levels, dtype=int)
        
        for z_idx in z_indices:
            z_val = self.z[z_idx]
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            
            # 绘制切片
            surf = ax.plot_surface(X, Y, 
                                 np.full_like(X, z_val),
                                 facecolors=plt.cm.jet(self.wind_speed[:, :, z_idx] / np.nanmax(self.wind_speed)),
                                 alpha=0.5)
            
            # 添加箭头
            skip = 5
            ax.quiver(X[::skip, ::skip], 
                     Y[::skip, ::skip],
                     np.full_like(X[::skip, ::skip], z_val),
                     self.u[::skip, ::skip, z_idx],
                     self.v[::skip, ::skip, z_idx],
                     np.zeros_like(X[::skip, ::skip]),
                     length=0.5, normalize=True, color='k', alpha=0.3)
        
        self.add_direction_arrows(ax)
        plt.colorbar(surf, label='Wind Speed (m/s)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.title('3D Wind Field - Slice Plot')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def particle_animation(self, output_path, n_particles=100, n_frames=100):
        """粒子动画可视化"""
        fig, ax = self.setup_3d_axes()
        
        # 初始化粒子位置
        particles = np.random.rand(n_particles, 3)
        particles[:, 0] = particles[:, 0] * (self.x[-1] - self.x[0]) + self.x[0]
        particles[:, 1] = particles[:, 1] * (self.y[-1] - self.y[0]) + self.y[0]
        particles[:, 2] = particles[:, 2] * (self.z[-1] - self.z[0]) + self.z[0]
        
        from scipy.interpolate import RegularGridInterpolator
        
        # 创建插值函数
        u_interp = RegularGridInterpolator((self.x, self.y, self.z), self.u)
        v_interp = RegularGridInterpolator((self.x, self.y, self.z), self.v)
        w_interp = RegularGridInterpolator((self.x, self.y, self.z), self.w)
        
        scatter = ax.scatter([], [], [], c='b', s=1)
        
        def update(frame):
            nonlocal particles
            
            # 更新粒子位置
            try:
                u_vel = u_interp(particles)
                v_vel = v_interp(particles)
                w_vel = w_interp(particles)
            except ValueError:
                # 如果粒子超出边界，重新初始化
                particles = np.random.rand(n_particles, 3)
                particles[:, 0] = particles[:, 0] * (self.x[-1] - self.x[0]) + self.x[0]
                particles[:, 1] = particles[:, 1] * (self.y[-1] - self.y[0]) + self.y[0]
                particles[:, 2] = particles[:, 2] * (self.z[-1] - self.z[0]) + self.z[0]
                u_vel = u_interp(particles)
                v_vel = v_interp(particles)
                w_vel = w_interp(particles)
            
            dt = 0.1
            particles[:, 0] += u_vel * dt
            particles[:, 1] += v_vel * dt
            particles[:, 2] += w_vel * dt
            
            scatter._offsets3d = (particles[:, 0], particles[:, 1], particles[:, 2])
            return scatter,
        
        self.add_direction_arrows(ax)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.title('3D Wind Field - Particle Animation')
        
        anim = animation.FuncAnimation(fig, update, frames=n_frames, 
                                     interval=50, blit=True)
        anim.save(output_path, writer='pillow')
        plt.close() 