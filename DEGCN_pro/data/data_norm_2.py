import numpy as np
from tqdm import tqdm

file_workdir = './data/'
label_file = ['train_label', 'test_A_label']
file = ['train_joint', 'test_A_joint', 'test_joint_B']  # 去0帧数据N,T,M,V,C
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
    joint_validframe[1] = np.array(data0['A_valid'])
    joint_validframe[2] = np.array(data0['B_valid'])   
    joint_data[0] = np.array(data0['x_train'])
    joint_data[1] = np.array(data0['A_x'])
    joint_data[2] = np.array(data0['B_x'])    
    label_data[0] = np.array(data0['y_train'])
    label_data[1] = np.array(data0['A_y'])   
    single_index[0] = np.array(data0['train_single'])
    single_index[1] = np.array(data0['A_single'])
    single_index[2] = np.array(data0['B_single'])   
    double_index[0] = np.array(data0['train_double'])
    double_index[1] = np.array(data0['A_double'])
    double_index[2] = np.array(data0['B_double'])    
    data0.close()

def rid_large_point():
    print('rid_large_point')
    thre = 800
    mask = np.ones(joint_data[0].shape[0], dtype=bool)
    for f in range(3):
        for i in range(joint_data[f].shape[0]):
            valid_frame = np.array([], dtype=int)
            
            for j in range(joint_validframe[f][i]):
                max = np.max(np.abs(joint_data[f][i][j,0]))
                if max <= thre:
                    valid_frame = np.concatenate((valid_frame, np.array([j], dtype=int)))

            if len(valid_frame) == 0:
                if f == 0:
                    joint_validframe[f][i] = 0
                    mask[i] = False
                    continue
                else:
                    index = np.argwhere(joint_data[f][i][:,0]>thre)
                    for t,v,c in index:
                        joint_data[f][i][t,0,v,c] = 0
            
            for j in range(joint_validframe[f][i]):
                if j not in valid_frame:
                    absolute_diff = np.abs(valid_frame - j)
                    closest_index = np.argmin(absolute_diff)
                    closest_frame = valid_frame[closest_index]
                    index = np.argwhere(np.abs(joint_data[f][i][j,0])>thre)
                    for v,c in index:
                        joint_data[f][i][j,0][v,c] = joint_data[f][i][closest_frame,0][v,c]
                    valid_frame = np.concatenate((valid_frame, np.array([j], dtype=int)))

        for i in double_index[f]:
            valid_frame = np.array([], dtype=int)
            
            for j in range(joint_validframe[f][i]):
                max = np.max(np.abs(joint_data[f][i][j,1]-joint_data[f][i][j,1][1]))
                if max <= thre:
                    valid_frame = np.concatenate((valid_frame, np.array([j], dtype=int)))
            
            if len(valid_frame) == 0:
                if f == 0:
                    joint_validframe[f][i] = 0
                    mask[i] = False
                    continue
                else:
                    index = np.argwhere(joint_data[f][i][:,1]>thre)
                    for t,v,c in index:
                        joint_data[f][i][t,1,v,c] = 0
            
            for j in range(joint_validframe[f][i]):
                if j not in valid_frame:
                    absolute_diff = np.abs(valid_frame - j)
                    closest_index = np.argmin(absolute_diff)
                    closest_frame = valid_frame[closest_index]
                    index = np.argwhere(np.abs(joint_data[f][i][j,1]-joint_data[f][i][j,1][1])>thre)
                    for v,c in index:
                        joint_data[f][i][j,1][v,c] = joint_data[f][i][closest_frame,1][v,c]
                    valid_frame = np.concatenate((valid_frame, np.array([j], dtype=int)))   

    joint_data[0] = joint_data[0][mask]
    label_data[0] = label_data[0][mask]
    joint_validframe[0] = joint_validframe[0][mask]

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

# def normalize():
#     for f in range(3):
#         joint_data[f] = joint_data[f].reshape(-1,300,34,3)
#         for i in range(joint_data[f].shape[0]):
#             joint_data[f][i] = joint_data[f][i] / np.abs(joint_data[f][i]).max()
def normalize():
    for f in range(3):
        joint_data[f] = joint_data[f].reshape(-1,300,34,3)
        for i in range(joint_data[f].shape[0]):
            mean_values = np.mean(joint_data[f][i], axis=(0,1))
            joint_data[f][i] = joint_data[f][i]-mean_values
            joint_data[f][i] = joint_data[f][i]/np.abs(joint_data[f][i]).max()
            
def save_data():
    print('save_data')
    file_save_name_A = file_workdir + 'data' + '.npz'
    file_save_name_B = file_workdir + 'B_data' + '.npz'

    x_tr = np.array(joint_data[0]).reshape(-1, 300, 34*3)
    print(x_tr.shape)
    A_te = np.array(joint_data[1]).reshape(-1, 300, 34*3)
    B_te = np.array(joint_data[2]).reshape(-1, 300, 34*3)
    
    y_tr = np.array(label_data[0])
    y_A = np.array(label_data[1])
    y_B = np.zeros((joint_data[2].shape[0], 155))
    y_B[:, 0] = 1
    
    np.savez(file_save_name_A, x_train=x_tr, x_test=A_te, y_train=y_tr, y_test=y_A)
    np.savez(file_save_name_B, x_train=x_tr, x_test=B_te, y_train=y_tr, y_test=y_B)

if __name__ == '__main__':
    load_fixed_data()
    rid_large_point() #去除大坐标噪点，可能无效
    length_denoise_sample() #长度阈值去样本
    normalize() #归一化
    save_data()
