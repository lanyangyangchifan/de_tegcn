import numpy as np

file_workdir = './data/'
label_file = ['train_label','test_A_label']
file = ['train_joint','test_A_joint','test_joint_B']#去0帧数据N,T,M,V,C
file_suffix = '.npy'
joint_validframe = [None] * 3
joint_data = [None] * 3
label_data = [None] * 2
single_index = [None] * 3 #单人动作标签
double_index = [None] * 3 #双人动作标签

def load_joint_data():
    print('load_joint_data')
    for i in range(3):
        file_name = file_workdir + file[i] + '_rid0' + file_suffix
        joint_data[i] = np.load(file_name)
        file_name = file_workdir + file[i] + '_validframe' + file_suffix
        joint_validframe[i] = np.load(file_name)

def load_label():
    print('load_label')
    for i in range(2):
        file_name = file_workdir + label_file[i] + file_suffix
        data = np.load(file_name)
        labels_vector = np.zeros((len(data), 155))
        for idx, l in enumerate(data):
            labels_vector[idx, l] = 1
            label_data[i] = labels_vector

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

def medium_filter():
    for f in range(3):
        for i in range(joint_data[f].shape[0]):
            for j in range(1,joint_validframe[f][i]-1):
                for m in range(2):
                    for v in range(17):
                        for c in range(3):
                            joint_data[f][i,j,m,v,c] = np.median(joint_data[f][i,j-1:j+2,m,v,c])

def find_index():
    print('find_index')
    for i in range(3):
        double_index[i] = np.where(np.any(joint_data[i][:, :, 1, :, :] != 0, axis=(1, 2, 3)))[0]
        single_index[i] = np.where(np.all(joint_data[i][:, :, 1, :, :] == 0, axis=(1, 2, 3)))[0]

def loc_denoise_sample():    # 双人错位修正
    print('loc_denoise_sample')
    for f in range(3):
        for i in double_index[f]:
            for j in range(joint_data[f].shape[1]-1):
                if j+1 == joint_validframe[f][i]:
                    break
                max_dis = max(np.sum(np.linalg.norm(joint_data[f][i, j+1, 0] - joint_data[f][i, j, 1], axis=-1)),\
                    np.sum(np.linalg.norm(joint_data[f][i, j+1, 0] - joint_data[f][i, j, 0], axis=-1)),\
                    np.sum(np.linalg.norm(joint_data[f][i, j+1, 1] - joint_data[f][i, j, 0], axis=-1)),\
                    np.sum(np.linalg.norm(joint_data[f][i, j+1, 1] - joint_data[f][i, j, 1], axis=-1)))
                
                if np.sum(np.linalg.norm(joint_data[f][i, j+1, 0] - joint_data[f][i, j, 0], axis=-1)) == max_dis\
                    or np.sum(np.linalg.norm(joint_data[f][i, j+1, 1] - joint_data[f][i, j, 1], axis=-1)) == max_dis:
                    temp = joint_data[f][i][j+1][0].copy()
                    joint_data[f][i][j+1][0] = joint_data[f][i][j+1][1].copy()
                    joint_data[f][i][j+1][1] = temp

def fix_joint1():
    print('fix_joint1')
    for f in range(3):
        for i in range (joint_data[f].shape[0]):
            frame = np.where(joint_data[f][i][:,0,1,0:3] != [0,0,0])[0]
            for j in frame: #N,T,M,V,C
                temp = joint_data[f][i,j,0,1].copy()
                for v in range(joint_data[f].shape[3]):
                    joint_data[f][i,j,0,v] = joint_data[f][i,j,0,v] - temp
                if i in double_index[f]:
                    for v in range(joint_data[f].shape[3]):
                        joint_data[f][i,j,1,v] = joint_data[f][i,j,1,v] - temp

def distance_correct():
    print('双人距离调整')
    for f in range(3):
        for i in double_index[f]:
            valid_frame = np.array([], dtype=int)
            
            for j in range(joint_validframe[f][i]):
                distance = np.linalg.norm(joint_data[f][i][j][0][1] - joint_data[f][i][j][1][1], ord=2)
                if distance < 10:
                    valid_frame = np.concatenate((valid_frame, np.array([j], dtype=int)))
            
            if len(valid_frame) == 0:
                joint_data[f][i][:, 1, :] += np.array([5, 5, 0], dtype=np.float32) - joint_data[f][i][j][1][1]
                continue
            
            for j in range(joint_validframe[f][i]):
                if j not in valid_frame:
                    absolute_diff = np.abs(valid_frame - j)
                    closest_index = np.argmin(absolute_diff)
                    closest_frame = valid_frame[closest_index]
                    joint_data[f][i][j, 1, :] += joint_data[f][i][closest_frame][1][1] - joint_data[f][i][j][1][1]
                    valid_frame = np.concatenate((valid_frame, np.array([j], dtype=int)))

def save_fixed_data():
    file_name = file_workdir + 'fixed_data' + '.npz'
    np.savez(file_name, x_train=joint_data[0], A_x=joint_data[1], B_x=joint_data[2],\
              y_train=label_data[0], A_y=label_data[1], train_valid=joint_validframe[0],\
                A_valid=joint_validframe[1], B_valid=joint_validframe[2],\
                    train_single = single_index[0], A_single = single_index[1], B_single = single_index[2],\
                        train_double = double_index[0], A_double = double_index[1], B_double = double_index[2],)

if __name__ == '__main__':
    load_joint_data() #加载节点数据
    load_label() #加载标签数据
    length_denoise_sample() #长度阈值去样本
    # medium_filter()
    find_index() #找单人双人索引
    loc_denoise_sample() #双人位置错位修正
    fix_joint1() #固定节点1
    # distance_correct() #双人距离过大修正，可能无效
    save_fixed_data()