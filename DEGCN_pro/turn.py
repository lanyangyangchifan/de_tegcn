import pickle
import numpy as np
workdir = './work_dir/jbf_5_test/'#注意修改文件夹
file_name = workdir + 'epoch1_test_score.pkl'
# 加载.pkl文件
with open(file_name, 'rb') as f:
    data_dict = pickle.load(f)

# 提取值并构建数组
value_list = list(data_dict.values())  # 提取字典的值
array_2d = np.array(value_list)  # 将值列表转换为NumPy数组
np.save(workdir + 'pred.npy', array_2d)
# 输出数组的形状
print("转换后的数组形状:", array_2d.shape)