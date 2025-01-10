import netCDF4 as nc
import numpy as np

class NCDataReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = nc.Dataset(file_path)
        
    def read_wind_data(self, time_step: int = 0):
        """读取风速数据和网格信息
        Args:
            time_step: 时间步长索引
        """
        # 读取三个方向的风速分量
        u = self.dataset.variables['u'][time_step]  # x方向风速 (zu_3d, y, xu)
        v = self.dataset.variables['v'][time_step]  # y方向风速 (zu_3d, yv, x)
        w = self.dataset.variables['w'][time_step]  # z方向风速 (zw_3d, y, x)
        
        # 读取网格信息
        x = self.dataset.variables['x'][:]   # x方向坐标
        xu = self.dataset.variables['xu'][:] # u分量的x方向坐标
        y = self.dataset.variables['y'][:]   # y方向坐标
        yv = self.dataset.variables['yv'][:] # v分量的y方向坐标
        zu = self.dataset.variables['zu_3d'][:] # u,v分量的垂直坐标
        zw = self.dataset.variables['zw_3d'][:] # w分量的垂直坐标
        
        # 打印数据维度
        print(f"数据维度信息:")
        print(f"u shape: {u.shape} (zu_3d, y, xu)")
        print(f"v shape: {v.shape} (zu_3d, yv, x)")
        print(f"w shape: {w.shape} (zw_3d, y, x)")
        print(f"\n坐标维度信息:")
        print(f"x: {x.shape}, xu: {xu.shape}")
        print(f"y: {y.shape}, yv: {yv.shape}")
        print(f"zu: {zu.shape}, zw: {zw.shape}")
        
        # 处理masked数组
        u = self._handle_masked_array(u)
        v = self._handle_masked_array(v)
        w = self._handle_masked_array(w)

        u = np.transpose(u, (0, 2, 1))
        v = np.transpose(v, (0, 2, 1))
        w = np.transpose(w, (0, 2, 1))
        
        # 将w插值到与u,v相同的垂直层
        if zu.shape != zw.shape:
            # 创建插值函数
            from scipy.interpolate import interp1d
            w_interp = np.zeros((zu.shape[0], w.shape[1], w.shape[2]))
            for i in range(w.shape[1]):
                for j in range(w.shape[2]):
                    f = interp1d(zw, w[:, i, j], bounds_error=False, fill_value="extrapolate")
                    w_interp[:, i, j] = f(zu)
            w = w_interp
        
        return xu, y, zu, u, v, w
    
    def _handle_masked_array(self, data):
        """处理masked数组，将masked的值设为0"""
        if isinstance(data, np.ma.MaskedArray):
            # 将masked的值设为0
            filled_data = data.filled(0)
            return filled_data
        return data
    
    def get_terrain_surface(self, time_step: int = 0):
        """获取地形表面网格点
        Args:
            time_step: 时间步长索引
        Returns:
            x, y, z: 地形表面的坐标点
        """
        # 获取u场的mask
        u = self.dataset.variables['u'][time_step]
        u = np.transpose(u, (0, 2, 1))
        if not isinstance(u, np.ma.MaskedArray):
            return None, None, None
            
        # 获取坐标
        x = self.dataset.variables['xu'][:]
        y = self.dataset.variables['y'][:]
        z = self.dataset.variables['zu_3d'][:]
        
        # 创建网格点
        xx, yy = np.meshgrid(x, y, indexing='ij')
        terrain_height = np.zeros_like(xx)
        
        # 对每个水平位置，找到最高的mask点作为地形高度
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                mask_profile = u.mask[:, j, i]  # 垂直剖面的mask
                if np.any(mask_profile):
                    # 找到最高的mask点的索引
                    highest_mask = np.where(mask_profile)[0][-1]
                    terrain_height[i, j] = z[highest_mask]
                    
        return xx, yy, terrain_height
        
    def get_terrain_mask(self):
        """获取地形mask"""
        # 使用u变量来获取地形mask
        u = self.dataset.variables['u'][0]
        u = np.transpose(u, (0, 2, 1))
        if isinstance(u, np.ma.MaskedArray):
            return u.mask
        return None
    
    def close(self):
        """关闭数据集"""
        self.dataset.close() 