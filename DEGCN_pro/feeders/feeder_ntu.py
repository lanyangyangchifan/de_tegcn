import numpy as np

from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False): # 定义初始化方法，初始化 Feeder 类的实例。方法的参数包括数据路径、标签路径、数据集划分方式、各种数据增强选项等
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute

        ：参数data_path：
        ：参数标签路径：
        ：param split：训练集或测试集
        ：param random_choose：如果为true，则随机选择输入序列的一部分
        ：param random_shift：如果为真，则在序列的开始或结束处随机填充零
        ：参数random_move：
        ：param random_rot：围绕xyz轴旋转骨架
        ：param window_size：输出序列的长度
        ：param normalization：如果为true，则规范化输入序列
        ：param debug：如果为true，则仅使用前100个样本
        ：param use_mmap：如果为true，则使用mmap模式加载数据，这可以节省运行内存
        ：param bone：是否使用骨骼模式
        ：param vel：是否使用运动模态
        ：param-only_label：仅加载用于集合分数计算的标签
        """

        # 将传入的参数赋值给实例变量，例如 self.data_path 存储数据路径，self.split 存储数据集划分（训练或测试）
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data() # 调用 load_data() 方法，加载数据
        if normalization: # 如果 normalization 为真，则调用 get_mean_map() 方法计算数据的均值和标准差，以便后续标准化处理
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train'] # self.data 存储训练数据，self.label 存储标签（只有当标签大于0时才保留），self.sample_name 用于生成样本名称
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape # 获取数据的维度，N 是样本数，T 是时间步数
        self.data = self.data.reshape((N, T, 2, 17, 3)).transpose(0, 4, 1, 3, 2) # 将数据重新调整形状为 (N, T, 2, 17, 3)，然后转置，使得数据的维度顺序变为 (N, 3, T, 17, 2)
        # self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self): # 定义计算均值和标准差的方法
        data = self.data # 简化引用，N 是样本数，C 是通道数，T 是时间步数，V 是关键点数，M 是额外的维度
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0) # 计算数据的均值，首先沿着时间轴（axis=2）取均值，然后沿着额外维度（axis=4）取均值，最后沿着通道维度（axis=0）取均值
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1)) # 先转置数据，然后将其重塑为 (N * T * M, C * V)，计算标准差，并将其调整为形状 (C, 1, V, 1)

    def __len__(self):
        return len(self.label) # 定义返回数据集长度的方法，返回标签的长度

    def __iter__(self):
        return self # 定义迭代器，使得 Feeder 类的实例可以被迭代

    def __getitem__(self, index): # 定义根据索引获取数据项的方法
        data_numpy = self.data[index] # 根据索引提取相应的数据和标签，并将数据转换为 NumPy 数组
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0) # 计算有效帧数量，检查数据的每一帧是否包含非零值
        if valid_frame_num == 0:
            valid_frame_num = 1
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size) # 使用 tools 模块中的 valid_crop_resize 函数处理数据，调整数据形状
        if self.random_rot: # 如果 random_rot 为真，则调用 tools.random_rot 方法随机旋转数据
            data_numpy = tools.random_rot(data_numpy)
        if self.bone: # 如果 bone 为真，计算骨骼数据，使用 ntu_pairs 中的配对信息
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel: # 如果 vel 为真，计算每一帧的速度（差分），最后一帧设为零
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index # 返回处理后的数据、标签和索引

    def top_k(self, score, top_k): # 定义计算 Top-K 准确率的方法
        rank = score.argsort() # 对分数进行排序并获取排序后的索引
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)] # 检查每个标签是否在 Top-K 排名中
        return sum(hit_top_k) * 1.0 / len(hit_top_k) # 返回 Top-K 准确率，计算命中数与总数的比例


def import_class(name): # 定义一个动态导入类的方法
    components = name.split('.') # 根据传入的类名字符串，将其分解为模块和类，逐层导入，最后返回类对象，可以在运行时动态加载类
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
