import torch
import numpy as np
#去全0帧,保存节点数组N,T,M,V,C
file_workdir = './data/'
file = ['train_joint','test_A_joint','test_joint_B'] #原始数据N,C,T,V,M
file_suffix = '.npy'

# 设置设备为GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rid_0():
    for i in range(3):
        print(f'process {i} dataset')
        file_name = file_workdir + file[i] + file_suffix
        file_save_name = file_workdir + file[i] + '_rid0' + file_suffix
        data0 = np.load(file_name)
        data = data0.transpose(0, 2, 4, 3, 1) #N,C,T,V,M->N,T,M,V,C       
        data = torch.tensor(data, dtype=torch.float32).to(device)# 转换 NumPy 数组为 PyTorch 张量并将数据移动到 GPU
        valid_frame_data = np.full(data.size(0), data.size(1))
        indices = torch.argwhere(torch.all(data == 0, dim=(2,3,4)))#找全0样本帧
        count_dict = {}#整合每个样本的全0帧
        for array in indices:
            first_num = array[0].item()
            if first_num in count_dict:
                count_dict[first_num] = torch.cat((count_dict[first_num], array[1].unsqueeze(0)))
            else:
                count_dict[first_num] = array[1].unsqueeze(0)
        zero_matrix = torch.zeros((1, data.size(2), data.size(3), data.size(4)), dtype=torch.float32).to(device)
        data_rid0 = None
        for sample in range(data.size(0)):
            one_sample = data[sample]
            if sample in count_dict:#去0帧
                valid_frame_data[sample] = data.size(1) - len(count_dict[sample])
                mask = torch.ones(data.size(1), dtype=torch.bool).to(device)
                mask[count_dict[sample]] = False
                one_sample = one_sample[mask]
                one_sample = torch.cat((one_sample, torch.tile(zero_matrix, (len(count_dict[sample]), 1, 1, 1))), dim=0)
            expanded_four_dim_array = one_sample.unsqueeze(0)#连接各样本
            if sample == 0:
                data_rid0 = expanded_four_dim_array
            else:
                data_rid0 = torch.cat((data_rid0, expanded_four_dim_array), dim=0) 
        data_rid0 = data_rid0.cpu().numpy()  
        print(data_rid0.shape)      
        np.save(file_save_name, data_rid0)   
        np.save(file_workdir + file[i] + '_validframe' + file_suffix, valid_frame_data)  

if __name__ == '__main__':
    rid_0()
