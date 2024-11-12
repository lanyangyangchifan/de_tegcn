import numpy as np
import pickle
import os

# 定义文件路径
work_dir = './'
npy_file1 = os.path.join(work_dir, 'new_train_label.npy')
npy_file2 = os.path.join(work_dir, 'val_label.npy')
npy_file3 = os.path.join(work_dir, 'labeled_test_label.npy')
pkl_file1 = os.path.join(work_dir, 'new_train_label.pkl')
pkl_file2 = os.path.join(work_dir, 'val_label.pkl')
pkl_file3 = os.path.join(work_dir, 'labeled_test_label.pkl')

def convert_npy_to_pkl(npy_file, pkl_file):
    """
    将 .npy 文件转换为 .pkl 文件，并为数据添加样本名称。
    """
    try:
        # 读取 .npy 文件
        data = np.load(npy_file, allow_pickle=True)
        print(f"成功加载 {npy_file}。")

        # 将数据保存为 .pkl 文件
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"{npy_file} 转换为 {pkl_file} 完成。")

        # 添加样本名称并保存为新的 .pkl 文件
        sample_names = [f'sample{i+1}' for i in range(len(data))]
        result_tuple = (tuple(sample_names), data.tolist())

        new_pkl_file = pkl_file.replace('.pkl', '_converted.pkl')
        with open(new_pkl_file, 'wb') as f:
            pickle.dump(result_tuple, f)
        print(f"{pkl_file} 转换并保存为 {new_pkl_file} 完成。")

    except FileNotFoundError:
        print(f"错误：未找到文件 {npy_file}")
    except Exception as e:
        print(f"发生错误：{e}")

# 执行转换操作
# convert_npy_to_pkl(npy_file1, pkl_file1)
# convert_npy_to_pkl(npy_file2, pkl_file2)
convert_npy_to_pkl(npy_file3, pkl_file3)
