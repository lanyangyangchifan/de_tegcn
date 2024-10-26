import torch
import numpy as np

# 设置设备为GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_workdir = './data/contest/'
label_file = ['train_label','test_A_label']
file = ['train_joint','test_A_joint','test_joint_B']
file_suffix = '.npy'
file_save_name_A = file_workdir + 'data' + '.npz'
file_save_name_B = file_workdir + 'B_data' + '.npz'

rid_0_data = [None] * 3 #去0帧结果N,T,M,V,C
label_data = [None] * 2 #标签数组

def load_label():
    for i in range(2):
        file_name = file_workdir + label_file[i] + file_suffix
        data = np.load(file_name)
        labels_vector = np.zeros((len(data), 155))
        print(len(data))
        for idx, l in enumerate(data):
            labels_vector[idx, l] = 1
            label_data[i] = labels_vector
        print(label_data[i].shape)

def rid_0():
    for i in range(3):
        print(f'process {i} dataset')
        file_name = file_workdir + file[i] + file_suffix
        # file_save_name = file_workdir + file[i] + '_rid0' + file_suffix
        data0 = np.load(file_name)
        data = data0.transpose(0, 2, 4, 3, 1) #N,C,T,V,M->N,T,M,V,C

        # 转换 NumPy 数组为 PyTorch 张量并将数据移动到 GPU
        data = torch.tensor(data, dtype=torch.float32).to(device)

        indices = torch.argwhere(torch.all(data == 0, dim=(2,3,4)))
        print(indices)

        # 创建一个空字典来存储每个数出现的次数
        count_dict = {}

        # 遍历每个数组，获取第一个数并更新计数
        for array in indices:
            first_num = array[0].item()
            # print(first_num)
            if first_num in count_dict:
                count_dict[first_num] = torch.cat((count_dict[first_num], array[1].unsqueeze(0)))
            else:
                count_dict[first_num] = array[1].unsqueeze(0)
        print(count_dict)
        # 去0补0
        zero_matrix = torch.zeros((1, data.size(2), data.size(3), data.size(4)), dtype=torch.float32).to(device)
        data_rid0 = None

        for sample in range(data.size(0)):
            one_sample = data[sample]
            if sample in count_dict:
                mask = torch.ones(data.size(1), dtype=torch.bool).to(device)
                mask[count_dict[sample]] = False
                one_sample = one_sample[mask]
                one_sample = torch.cat((one_sample, torch.tile(zero_matrix, (len(count_dict[sample]), 1, 1, 1))), dim=0)
            expanded_four_dim_array = one_sample.unsqueeze(0)
            if sample == 0:
                data_rid0 = expanded_four_dim_array
            else:
                data_rid0 = torch.cat((data_rid0, expanded_four_dim_array), dim=0)
            # print(data_rid0.size())

        # 将数据转换为 NumPy 数组并保存
        data_rid0 = data_rid0.cpu().numpy()
        rid_0_data[i] = data_rid0
        # np.save(file_save_name, data_rid0)    

def rid_0_sample():
    x_train = rid_0_data[0]
    sample_0_index = np.where(np.all(x_train == 0, axis=(1,2,3,4)))#全0样本
    repeat_index = []
    for i in range(x_train.shape[0]):
        if np.array_equal(x_train[i,0],x_train[i,1]) and np.array_equal(x_train[i,2],x_train[i,1]):
            repeat_index.append(i)
    print(sample_0_index)
    print(repeat_index)
    # 找出满足条件的索引

    # 使用布尔索引创建新的数组
    x_train_mask = np.ones(x_train.shape[0], dtype=bool)
    x_train_mask[sample_0_index] = False
    x_train_mask[repeat_index] = False

    rid_0_data[0] = x_train[x_train_mask]
    label_data[0] = label_data[0][x_train_mask]
    print(rid_0_data[0].shape)

def length_denoise_sample():
    valid_frame_thre = 20 #11
    x_train = rid_0_data[0]
    mask = np.ones(x_train.shape[0], dtype=bool)
    N = x_train.shape[0]
    for i in range(N):
        for j in range(valid_frame_thre):
            one_frame = x_train[i][j]
            if np.sum(one_frame) == 0: #有效帧数<valid_frame_thre
                mask[i] = False
                print(i)
                break
    rid_0_data[0] = x_train[mask]
    label_data[0] = label_data[0][mask]
    print(rid_0_data[0].shape)

def loc_denoise_sample():    # 双人错位修正
    for f in range(3):
        indices = np.where(np.any(rid_0_data[f][:, :, 1, :, :] != 0, axis=(1, 2, 3)))[0]
        for i in indices:
            for j in range(rid_0_data[f].shape[1]-1):
                if np.all(rid_0_data[f][i, j+1:, :, :, :] == 0):
                    break
                max_dis = max(np.sum(np.linalg.norm(rid_0_data[f][i, j+1, 0] - rid_0_data[f][i, j, 1], axis=-1)),\
                    np.sum(np.linalg.norm(rid_0_data[f][i, j+1, 0] - rid_0_data[f][i, j, 0], axis=-1)),\
                    np.sum(np.linalg.norm(rid_0_data[f][i, j+1, 1] - rid_0_data[f][i, j, 0], axis=-1)),\
                    np.sum(np.linalg.norm(rid_0_data[f][i, j+1, 1] - rid_0_data[f][i, j, 1], axis=-1)))
                
                if np.sum(np.linalg.norm(rid_0_data[f][i, j+1, 0] - rid_0_data[f][i, j, 0], axis=-1)) == max_dis\
                    or np.sum(np.linalg.norm(rid_0_data[f][i, j+1, 1] - rid_0_data[f][i, j, 1], axis=-1)) == max_dis:
                    print(i,j)
                    temp = rid_0_data[f][i][j+1][0].copy()
                    # print(temp)
                    rid_0_data[f][i][j+1][0] = rid_0_data[f][i][j+1][1].copy()
                    # print(temp)
                    rid_0_data[f][i][j+1][1] = temp

def fix_joint1():
    for f in range(3):
        for i in range (rid_0_data[f].shape[0]):
            frame = np.where(rid_0_data[f][i][:,0,1,0:3] != [0,0,0])[0]
            for j in frame:
                temp = rid_0_data[f][i,j,0,1].copy()
                for v in range(rid_0_data[f].shape[3]):
                    rid_0_data[f][i,j,0,v] = rid_0_data[f][i,j,0,v] - temp
                if not np.all(rid_0_data[f][i,j,1] == 0):
                    for v in range(rid_0_data[f].shape[3]):
                        rid_0_data[f][i,j,1,v] = rid_0_data[f][i,j,1,v] - temp

def normalize():
    for f in range(3):
        rid_0_data[f] = rid_0_data[f].reshape(-1,300,34,3)
        for i in range(rid_0_data[f].shape[0]):
            mean_values = np.mean(rid_0_data[f][i], axis=(0,1))
            rid_0_data[f][i] = rid_0_data[f][i]-mean_values
            rid_0_data[f][i] = rid_0_data[f][i]/np.abs(rid_0_data[f][i]).max()

def save_data():
    x_tr = rid_0_data[0].reshape(-1,300,34*3)
    print(x_tr.shape)
    A_te = rid_0_data[1].reshape(-1,300,34*3)
    print(A_te.shape)
    B_te = rid_0_data[2].reshape(-1,300,34*3)
    print(B_te.shape)
    y_tr = label_data[0]
    print(y_tr.shape)
    y_A = label_data[1]
    print(y_A.shape)
    y_B = np.zeros((rid_0_data[2].shape[0], 155))
    y_B[:,0] = 1
    print(y_B.shape)
    np.savez(file_save_name_A, x_train=x_tr, x_test=A_te, y_train=y_tr, y_test=y_A)
    np.savez(file_save_name_B, x_train=x_tr, x_test=B_te, y_train=y_tr, y_test=y_B)

if __name__ == '__main__':
    load_label()
    rid_0()
    rid_0_sample()
    length_denoise_sample()
    loc_denoise_sample()
    fix_joint1()
    normalize()
    save_data()