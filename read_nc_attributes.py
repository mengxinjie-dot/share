from netCDF4 import Dataset

# 打开.nc文件（可写模式）
file_path = '/home/mxj/docker_palm/share/tongren_dynamic'
file_path2 = '/home/mxj/docker_palm/share/wrf4palm_tongren_dynamic_2023_6_8_0'
data = Dataset(file_path, 'r')
data2 = Dataset(file_path2, 'r')

# 打印全局属性
print("Global Attributes:")
for attr in data.ncattrs():
    print(f"{attr}: {data.getncattr(attr)}")


# 保存并关闭文件
data.close()
data2.close()
