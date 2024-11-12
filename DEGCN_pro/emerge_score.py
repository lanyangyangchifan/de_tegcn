import numpy as np
import pickle

file_num = 10
# 'val' 'test' 
mode = 'test' #选择生成测试集置信度或评估集融合准确率

file_name = [None] * file_num
array = [None] * file_num
weight = [None] * file_num
weight = np.array(weight)

if mode == 'val':
    save_name = './data/val_data.npz'
    data0 = np.load(save_name)
    labels = data0['y_test']
    file_name[0] = './work_dir/result0\epoch1_test_score.pkl'
    file_name[1] = './work_dir/result1\epoch1_test_score.pkl'
    file_name[2] = './work_dir/result2\epoch1_test_score.pkl'
    file_name[3] = './work_dir/result3\epoch1_test_score.pkl'
    file_name[4] = './work_dir/result4\epoch1_test_score.pkl'
    file_name[5] = './work_dir/result5\epoch1_test_score.pkl'
    file_name[6] = './work_dir/result6\epoch1_test_score.pkl'
    file_name[7] = './work_dir/result7/val(43.2).pkl'
    file_name[8] = './work_dir/result8/val41.70.pkl'
    file_name[9] = './work_dir/result9/val44.15.pkl'

elif mode == 'test':
    file_name[0] = './work_dir/result0/test\epoch1_test_score.pkl'
    file_name[1] = './work_dir/result1/test\epoch1_test_score.pkl'
    file_name[2] = './work_dir/result2/test\epoch1_test_score.pkl'
    file_name[3] = './work_dir/result3/test\epoch1_test_score.pkl'
    file_name[4] = './work_dir/result4/test\epoch1_test_score.pkl'
    file_name[5] = './work_dir/result5/test\epoch1_test_score.pkl'
    file_name[6] = './work_dir/result6/test\epoch1_test_score.pkl'
    file_name[7] = './work_dir/result7/test/new_test_bone(43.2).pkl'
    file_name[8] = './work_dir/result8/test/test41.70.pkl'
    file_name[9] = './work_dir/result9/test/test44.15.pkl'

# 计算加权平均数组
weight[0] = 10# jbf归一化
weight[1] = 5# jbf归一化
weight[2] = 10# jbf归一化
weight[3] = 10# jbf归一化
weight[4] = 8# jbf归一化
weight[5] = 30# degcn归一化
weight[6] = 15 #degcn归一化
weight[7] = 25# tegcn归一化
weight[8] = 15# tegcn归一化
weight[9] = 90# tegcn原始数据

weight = weight.astype(float)

for i in range(file_num):
    with open(file_name[i], 'rb') as file:
        data_dict = pickle.load(file)
        value_list = list(data_dict.values())  # 提取字典的值
        array[i] = np.array(value_list)
array = np.array(array)

sum = np.sum(weight)
print(sum)

for i in range(file_num):
    weight[i] = weight[i]/sum

weighted_average = weight[0] * array[0]
for i in range(1,file_num):
    weighted_average += weight[i] * array[i]
print(weighted_average.shape)
if mode == 'test':
    np.save('./work_dir/emerge/pred.npy', weighted_average)
elif mode == 'val':
    num_same_positions = np.sum(np.argmax(weighted_average, axis=1) == np.argmax(labels, axis=1))
    percentage = num_same_positions / 2000 * 100
    print('分数：', percentage)

