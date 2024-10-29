import numpy as np
import pickle

# file_name1 = './work_dir/test/test_A/75_85_test_score.pkl'
# file_name2 = './work_dir/test/test_A/76_65_test_score.pkl'
# file_name3 = './work_dir/test/test_A/74_95_test_score.pkl'
file_name1 = './work_dir/test/test_B/75_85_test_score.pkl'
file_name3 = './work_dir/test/test_B/76_65_test_score.pkl'
file_name2 = './work_dir/test/test_B/74_95_test_score.pkl'
with open(file_name1, 'rb') as file1, open(file_name2, 'rb') as file2, open(file_name3, 'rb') as file3:
    data_dict1 = pickle.load(file1)
    data_dict2 = pickle.load(file2)
    data_dict3 = pickle.load(file3)

value_list1 = list(data_dict1.values())  # 提取字典的值
array1 = np.array(value_list1)
value_list2 = list(data_dict2.values())  # 提取字典的值
array2 = np.array(value_list2)
value_list3 = list(data_dict3.values())  # 提取字典的值
array3 = np.array(value_list3)

# 计算加权平均数组
weight1 = 3/11
weight2 = 4/11
weight3 = 4/11
weighted_average = weight1 * array1 + weight2 * array2 + weight3 * array3 + weight3 * array3 

#以下两个代码块根据需要注释一个

# 测试集B保存置信度文件
np.save('./work_dir/emerge/pred.npy', weighted_average)
print(weighted_average.shape)

# 测试集A打印综合准确率
# save_name = './data/data.npz'
# data0 = np.load(save_name)
# labels = data0['y_test']
# num_same_positions = np.sum(np.argmax(weighted_average, axis=1) == np.argmax(labels, axis=1))
# percentage = num_same_positions / 2000 * 100
# print(weight1, weight2, weight3, "相同位置的百分比：", percentage)
