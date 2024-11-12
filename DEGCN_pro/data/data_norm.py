import numpy as np
from tqdm import tqdm

file_workdir = './data/'
label_file = ['train_label','val_label']
file = ['train_joint','test_joint','val_joint']  # 去0帧数据N,T,M,V,C
file_suffix = '.npy'
joint_validframe = [None] * 3
joint_data = [None] * 3
label_data = [None] * 2
single_index = [None] * 3  # 单人动作标签
double_index = [None] * 3  # 双人动作标签

def load_fixed_data():
    print('load_fixed_data')
    data0 = np.load(file_workdir + 'fixed_data.npz')
    joint_validframe[0] = np.array(data0['train_valid'])
    joint_validframe[1] = np.array(data0['test_valid'])
    joint_validframe[2] = np.array(data0['val_valid'])   
    joint_data[0] = np.array(data0['x_train'])
    joint_data[1] = np.array(data0['x_test'])
    joint_data[2] = np.array(data0['x_val'])    
    label_data[0] = np.array(data0['y_train'])
    label_data[1] = np.array(data0['y_test'])   
    single_index[0] = np.array(data0['train_single'])
    single_index[1] = np.array(data0['test_single'])
    single_index[2] = np.array(data0['val_single'])   
    double_index[0] = np.array(data0['train_double'])
    double_index[1] = np.array(data0['test_double'])
    double_index[2] = np.array(data0['val_double'])    
    data0.close()

def length_denoise_sample():
    print('length_denoise_sample')
    valid_frame_thre = 20 #11
    mask = np.ones(len(joint_validframe[0]), dtype=bool)
    N = len(joint_validframe[0])
    for i in range(N):
        if joint_validframe[0][i] < valid_frame_thre:
            mask[i] = False
    joint_data[0] = joint_data[0][mask]
    label_data[0] = label_data[0][mask]
    joint_validframe[0] = joint_validframe[0][mask]

def find_index():
    print('find_index')
    for i in range(3):
        double_index[i] = np.where(np.any(joint_data[i][:, :, 1, :, :] != 0, axis=(1, 2, 3)))[0]
        single_index[i] = np.where(np.all(joint_data[i][:, :, 1, :, :] == 0, axis=(1, 2, 3)))[0]

def normalize():
    print('normalize')
    for f in range(3):
        joint_data[f] = joint_data[f].reshape(-1,300,34,3)
        for i in tqdm(range(joint_data[f].shape[0])):
            if joint_validframe[f][i] == 0:
                continue
            mean_values = np.mean(joint_data[f][i], axis=(0,1))
            joint_data[f][i] = joint_data[f][i]-mean_values
            joint_data[f][i] = joint_data[f][i]/np.abs(joint_data[f][i]).max()

def save_data():
    print('save_data')
    file_save_name_A = file_workdir + 'val_data' + '.npz'
    file_save_name_B = file_workdir + 'test_data' + '.npz'

    x_tr = np.array(joint_data[0]).reshape(-1, 300, 34*3)
    print(x_tr.shape)
    x_te = np.array(joint_data[1]).reshape(-1, 300, 34*3)
    x_va = np.array(joint_data[2]).reshape(-1, 300, 34*3)
    
    y_tr = np.array(label_data[0])
    y_va = np.array(label_data[1])
    y_te = np.zeros((joint_data[1].shape[0], 155))
    y_te[:, 0] = 1
    
    np.savez(file_save_name_A, x_train=x_tr, x_test=x_va, y_train=y_tr, y_test=y_va)
    np.savez(file_save_name_B, x_train=x_tr, x_test=x_te, y_train=y_tr, y_test=y_te)

if __name__ == '__main__':
    load_fixed_data()
    # rid_large_point() #去除大坐标噪点，可能无效
    # length_denoise_sample() #长度阈值去样本
    # find_index()
    normalize() #归一化
    save_data()
