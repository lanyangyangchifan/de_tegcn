from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from time import strftime, localtime
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import warnings
import re

import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

import thop
from copy import deepcopy

#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
import psutil
open_files = psutil.Process().open_files()
print(f"当前打开的文件数: {len(open_files)}")
#import apex

def init_seed(seed):    #用于初始化随机种子
    torch.cuda.manual_seed_all(seed)    #设置所有 GPU 的随机种子，使得后续在 GPU 上的随机操作可复现
    torch.manual_seed(seed)    #设置 CPU 上的随机种子，使得后续在 CPU 上的随机操作可复现
    np.random.seed(seed)    #为 NumPy 库设置随机种子，确保 NumPy 的随机操作可复现
    random.seed(seed)    #设置 Python 内置 random 模块的随机种子，确保使用该模块的随机操作可复现
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True

def get_current_timestamp():    #用于获取当前的时间戳
    ct = time.time()    #获取当前时间（以秒为单位的浮点数）
    ms = int((ct - int(ct)) * 1000)    #计算当前时间的毫秒部分
    return '[ {},{:0>3d} ] '.format(strftime('%Y-%m-%d %H:%M:%S', localtime(ct)), ms)    #格式化并返回当前时间的字符串，包含年月日时分秒和毫秒

def str2bool(v):    #用于将字符串转换为布尔值
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_parser():    #用于创建一个命令行参数解析器
    parser = argparse.ArgumentParser(    #创建一个 ArgumentParser 对象，描述该程序的功能
        description='Spatial Temporal Graph Convolution Network')

    parser.add_argument(    #添加工作目录参数 --work-dir，指定工作目录的路径，默认值为 ./work_dir/temp
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default='')    #添加模型保存名称参数 -model_saved_name，用于指定保存模型的名称，默认为空
    parser.add_argument(    #添加配置文件参数 --config，指定配置文件的路径，默认值为 ./config/nturgbd120-cross-subject/default.yaml
        '--config',
        # default='./config/nturgbd120-cross-subject/default.yaml',
        default='./config/default.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(    #添加训练或测试阶段参数 --phase，指定是训练还是测试，默认值为 train
        '--phase', default='train', help='must be train or test')
    parser.add_argument(    #添加保存评分参数 --save-score，用于指定是否保存分类评分，默认值为 True
        '--save-score',
        type=str2bool,
        default=True,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(    #添加随机种子参数 --seed，指定随机种子，默认值为 1
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(    #添加日志间隔参数 --log-interval，指定打印日志的间隔（以迭代次数为单位），默认值为 100
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(    #添加模型保存间隔参数 --save-interval，指定保存模型的间隔（以迭代次数为单位），默认值为 1
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(    #添加保存模型起始轮次参数 --save-epoch，指定保存模型的起始轮次，默认值为 0
        '--save-epoch',
        type=int,
        default=0,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(    #添加评估间隔参数 --eval-interval，指定评估模型的间隔（以迭代次数为单位），默认值为 1
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(    #添加打印日志参数 --print-log，指定是否打印日志，默认值为 True
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(    #添加显示 Top K 准确率参数 --show-topk，指定要显示的 Top K 准确率，默认值为 [1, 5]
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(    #添加数据加载器参数 --feeder，指定使用的数据加载器，默认值为 'feeder.feeder'
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(    #添加数据加载器工作线程数参数 --num-worker，指定数据加载器的工作线程数，默认值为 0
        '--num-worker',
        type=int,
        default=0,
        help='the number of worker for data loader')
    parser.add_argument(    #添加训练数据加载器参数 --train-feeder-args，指定训练数据加载器的参数，默认值为空字典
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(    #添加测试数据加载器参数 --test-feeder-args，指定测试数据加载器的参数，默认值为空字典
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')    #添加模型参数 --model，指定使用的模型，默认值为 None
    parser.add_argument(    #添加模型参数字典参数 --model-args，指定模型的参数字典，默认值为空字典
        '--model-args',
        default=dict(),
        help='the arguments of model')
    parser.add_argument(    #添加网络初始化权重参数 --weights，指定网络初始化所需的权重，默认值为 None
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(    #添加忽略权重参数 --ignore-weights，指定在初始化时要忽略的权重名称，默认值为空列表
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(    #添加优化器参数 --base-lr，指定初始学习率，默认值为 0.01
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(    #添加学习率调整步长参数 --step，指定学习率调整的轮次，默认为 [20, 40, 60]
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(    #添加设备参数 --device，指定用于训练或测试的 GPU 索引，默认值为 0
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')    #添加优化器类型参数 --optimizer，指定优化器的类型，默认值为 'SGD'
    parser.add_argument(    #添加 Nesterov 参数 --nesterov，指定是否使用 Nesterov 加速，默认值为 False
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(    #添加批量大小参数 --batch-size，指定训练时的批量大小，默认值为 256
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(    #添加测试批量大小参数 --test-batch-size，指定测试时的批量大小，默认值为 256
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(    #添加起始轮次参数 --start-epoch，指定从哪个轮次开始训练，默认值为 0
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(    #添加总轮次参数 --num-epoch，指定训练的总轮次，默认值为 80
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(    #添加权重衰减参数 --weight-decay，指定优化器的权重衰减，默认值为 0.0005
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(    #添加预热轮次参数 --warm_up_epoch，指定预热阶段的轮次，默认值为 0
        '--warm_up_epoch', 
        type=int, 
        default=0)
    parser.add_argument(    #添加余弦学习率调整轮次参数 --cosine_epoch，指定使用余弦学习率调整的轮次，默认值为 0
        '--cosine_epoch', 
        type=int, 
        default=0)
    parser.add_argument(    #添加半周期学习率参数 --half，指定是否使用半周期学习率，默认值为 True
        '--half', 
        type=str2bool, 
        default=True)
    return parser    #返回配置好的参数解析器对象，以便在后续的程序中使用

#该类实现了标签平滑交叉熵损失，旨在提高模型的鲁棒性，减少过拟合，同时提供了灵活的损失计算方式（求和、求平均）
class LabelSmoothingCrossEntropy(nn.Module):    #定义一个名为 LabelSmoothingCrossEntropy 的类，继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类
    def __init__(self, eps=0.1, reduction='mean'):    #定义初始化方法，接受两个参数：eps（平滑因子，默认值为 0.1）和 reduction（损失计算方式，默认为 'mean'）
        super(LabelSmoothingCrossEntropy, self).__init__()    #调用父类（nn.Module）的初始化方法，以确保父类正确初始化
        self.eps = eps    #将传入的 eps 参数存储为实例变量，用于后续的标签平滑计算
        self.reduction = reduction    #将传入的 reduction 参数存储为实例变量，决定损失计算的方式（例如：求和或求平均）

    def forward(self, output, target):    #定义前向传播方法，接受两个参数：output（模型的预测输出）和 target（真实标签）
        c = output.size()[-1]    #获取 output 的最后一个维度的大小，通常对应于类别数 c
        log_preds = F.log_softmax(output, dim=-1)    #对 output 进行 log softmax 操作，得到每个类别的对数概率分布，dim=-1 表示在最后一个维度上进行操作
        if self.reduction == 'sum':    #如果 reduction 设置为 'sum'，则计算所有类别的对数概率和的负值，得到总损失
            loss = -log_preds.sum()
        else:    #如果 reduction 不是 'sum'，则计算每个样本的对数概率和的负值，得到每个样本的损失
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':    #如果 reduction 设置为 'mean'，则对所有样本的损失取平均值
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)
        #返回最终损失：
        #第一部分 loss * self.eps / c 是平滑处理的损失，表示加权的标签平滑损失。
        #第二部分 (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction) 是标准的负对数似然损失（NLL）。这里使用 F.nll_loss 计算基于 log_preds 和 target 的损失。
        #将两部分损失加在一起，得到最终的损失值，结合了标签平滑和标准损失的优点。
    
class Processor():    #定义一个名为 Processor 的类，用于处理与基于骨骼的动作识别相关的任务
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):    #定义初始化方法，接受参数 arg，通常包含各种配置和参数
        self.arg = arg    #将传入的 arg 参数存储为实例变量，以便在类的其他方法中使用
        self.save_arg()    #调用 save_arg 方法，保存当前的配置参数
        self.ctime = ''.join(re.split('-|:|\[|\]', get_current_timestamp())).split(',')[0]    #获取当前时间戳，并将其格式化为字符串，作为后续保存路径的一部分
        self.savepath = self.arg.work_dir + '/' + self.ctime    #生成一个保存路径，结合工作目录和当前时间戳
        if not os.path.exists(self.savepath):    #检查保存路径是否存在，如果不存在则创建该目录
            os.makedirs(self.savepath)
        if arg.phase == 'train':    #检查当前阶段是否为训练阶段
            if not arg.train_feeder_args['debug']:    #如果不是调试模式，执行以下代码块
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')    #设置模型保存路径为工作目录下的 runs 文件夹
                if os.path.isdir(arg.model_saved_name):    #检查该路径是否已存在
                    print('log_dir: ', arg.model_saved_name, 'already exist')    #打印消息，告知用户该目录已存在
                    # answer = input('delete it? y/n:')    #提示用户是否删除该目录
                    # if answer == 'y':
                    #     shutil.rmtree(arg.model_saved_name)    #使用 shutil.rmtree 删除该目录及其内容
                    #     print('Dir removed: ', arg.model_saved_name)
                    # else:
                    #     print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')    #创建 TensorBoard 的训练和验证日志写入器，分别用于记录训练和验证过程中的信息
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')    #如果是调试模式，则将训练和验证的写入器都指向 test 文件夹
        self.global_step = 0    #初始化全局步数计数器
        self.load_model()    #调用 load_model 方法加载模型
        
        self.model = self.model.cuda(self.output_device)    #将模型移动到指定的 GPU 设备上

        if self.arg.phase == 'model_size':    #如果当前阶段是 model_size，则不做任何操作，直接跳过
            pass
        else:    #否则，加载优化器和数据
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr    #设置学习率为基本学习率
        self.best_acc = 0    #初始化最佳准确率和最佳准确率对应的训练轮数
        self.best_acc_epoch = 0

        if self.arg.half:    #检查是否使用半精度训练
            self.print_log('Use Half Traning!')

#            # 创建一个 GradScaler 实例，用于动态缩放梯度
#            self.scaler = torch.cuda.amp.GradScaler()
#        else:
#            self.scaler = None  # 如果不使用半精度，则不需要 GradScaler

#        # 使用 DataParallel 处理多个 GPU
#        if type(self.arg.device) is list:    
#            if len(self.arg.device) > 1:    
#                self.model = nn.DataParallel(
#                    self.model,
#                    device_ids=self.arg.device,
#                    output_device=self.output_device
#                )

#            self.model, self.optimizer = apex.amp.initialize(    #使用 Apex 库初始化模型和优化器，以支持混合精度训练
#                self.model,
#                self.optimizer,
#                opt_level='O1'
#            )
        
        else:
            if type(self.arg.device) is list:    #检查设备参数是否为列表
                if len(self.arg.device) > 1:    #如果设备列表中有多个设备，使用 DataParallel 将模型并行化，以支持多 GPU 训练
                    self.model = nn.DataParallel(
                        self.model,
                        device_ids=self.arg.device,
                        output_device=self.output_device)
        

    def load_data(self):    #定义 load_data 方法，用于加载训练和测试数据
        Feeder = import_class(self.arg.feeder)    #动态导入数据加载类，使用 import_class 函数
        self.data_loader = dict()    #初始化一个字典，用于存储数据加载器
        if self.arg.phase == 'train':    #检查当前阶段是否为训练阶段
            self.data_loader['train'] = torch.utils.data.DataLoader(    #创建训练数据加载器，并将其存储在字典中
                dataset=Feeder(**self.arg.train_feeder_args),    #使用 Feeder 类加载训练数据，传入训练参数
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
                #设置批量大小、是否打乱数据、工作线程数、是否丢弃最后一个不完整批次，并设置工作线程的初始化函数
        self.data_loader['test'] = torch.utils.data.DataLoader(    #创建测试数据加载器，并将其存储在字典中
            dataset=Feeder(**self.arg.test_feeder_args),    #使用 Feeder 类加载测试数据，传入测试参数
            batch_size=self.arg.test_batch_size,    #设置测试数据加载器的参数
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):    #定义 load_model 方法，用于加载模型
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device    #获取输出设备，如果设备参数是列表，则选择第一个设备
        self.output_device = output_device    #将输出设备存储为实例变量
        Model = import_class(self.arg.model)    #动态导入模型类，使用 import_class 函数
        shutil.copy2(inspect.getfile(Model), self.savepath)    #复制模型文件到保存路径，以备后续参考
        print(Model)
        self.model = Model(**self.arg.model_args)    #实例化模型，传入模型参数
        self.loss = LabelSmoothingCrossEntropy().cuda(output_device)    #实例化标签平滑交叉熵损失函数，并将其移动到输出设备

        if self.arg.weights:    #检查是否提供了权重文件
            self.global_step = int(arg.weights[:-3].split('_')[-1])    #从权重文件名中提取全局步数
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:    #检查权重文件格式
                with open(self.arg.weights, 'r') as f:    #如果是 .pkl 格式，使用 pickle 加载权重
                    weights = pickle.load(f)
            else:    #否则，使用 torch.load 加载权重
                weights = torch.load(self.arg.weights)

            #创建有序字典，将权重中 'module.' 前缀去掉，并将权重移动到输出设备
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())    #获取权重字典的所有键
            for w in self.arg.ignore_weights:    #遍历需要忽略的权重
                for key in keys:    #对每个权重键进行检查
                    if w in key:    #检查权重键中是否包含需要忽略的名称
                        if weights.pop(key, None) is not None:    #如果找到，尝试从权重字典中删除该键
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:    #尝试加载权重到模型
                self.model.load_state_dict(weights)
            except:    #加载失败
                state = self.model.state_dict()    #获取当前模型的状态字典（包含模型的所有参数）
                diff = list(set(state.keys()).difference(set(weights.keys())))    #找出当前状态字典中与加载权重字典不同的键
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)    #将加载的权重更新到模型的状态字典中
                self.model.load_state_dict(state)    #将更新后的状态字典加载到模型中

    def load_optimizer(self):    #定义 load_optimizer 方法，用于加载优化器
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(    #初始化 SGD 优化器，设置学习率、动量、Nesterov 加速和权重衰减
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(    #初始化 Adam 优化器，设置学习率和权重衰减
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'RMSProp':
            self.optimizer = optim.RMSprop(    #初始化 RMSProp 优化器，设置学习率、衰减因子和权重衰减
                self.model.parameters(), 
                lr=self.arg.base_lr, 
                alpha=0.9, 
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()    #如果优化器类型不匹配，抛出值错误
        
        #打印消息，告知用户正在使用预热策略，并显示预热的轮数
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):    #定义 save_arg 方法，用于保存参数
        # save arg
        arg_dict = vars(self.arg)    #将 arg 转换为字典格式，以便进行保存
        if not os.path.exists(self.arg.work_dir):    #检查工作目录是否存在，如果不存在则创建该目录
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:    #在工作目录下创建或打开 config.yaml 文件以进行写入
            f.write('# command line: {}\n\n'.format(' '.join(sys.argv)))    #在文件中写入当前的命令行参数
            yaml.dump(arg_dict, f)    #将参数字典以 YAML 格式写入文件中

    #     def adjust_learning_rate(self, epoch): #step
#         if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
#             if epoch < self.arg.warm_up_epoch:
#                 lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
#             else:
#                 lr = self.arg.base_lr * (0.1 ** np.sum(epoch >= np.array(self.arg.step)))
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] = lr
#             return lr
#         else:
#             raise ValueError()

#     def adjust_learning_rate(self, epoch): #cosine
#         if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
#             if epoch < self.arg.warm_up_epoch:
#                 lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
#             else:
#                 lr = self.arg.base_lr * (0.5 * (np.cos((epoch-self.arg.warm_up_epoch) / (self.arg.num_epoch-self.arg.warm_up_epoch) * np.pi) + 1))
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] = lr
#             return lr
#         else:
#             raise ValueError()

    def adjust_learning_rate(self, epoch):    #定义一个方法 adjust_learning_rate，接受参数 epoch，表示当前的训练轮次
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            num_epoch_ = self.arg.cosine_epoch + self.arg.warm_up_epoch    #计算总的训练轮数 num_epoch_，即余弦退火的轮数加上预热的轮数
            lr_cos = self.arg.base_lr * (0.5 * (np.cos((epoch-self.arg.warm_up_epoch) / (num_epoch_-self.arg.warm_up_epoch) * np.pi) + 1))    #计算余弦调度的学习率 lr_cos，基于当前轮数和总轮数的比例，使用余弦函数进行调整
            if epoch < self.arg.warm_up_epoch:    #当前轮数在预热阶段
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch    #按比例增加学习率，逐步提高到基础学习率
            elif epoch < num_epoch_ and lr_cos > 0.01:    #如果当前轮数在余弦调度阶段，且计算出的学习率大于 0.01，则使用 lr_cos
                lr = lr_cos
            else:    #否则，根据设置的学习率衰减步数降低学习率
                lr = self.arg.base_lr * (0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:    #更新优化器中所有参数组的学习率
                param_group['lr'] = lr
            return lr    #返回当前的学习率
        else:    #如果优化器不是 SGD 或 Adam，抛出值错误
            raise ValueError()

    def print_time(self):    #定义一个方法 print_time，用于打印当前时间
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):    #定义一个方法 print_log，接受要打印的字符串 str 和一个布尔值 print_time，用于控制是否打印时间
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.savepath), 'a') as f:    #打开一个日志文件 log.txt，以追加模式写入
                print(str, file=f)

    def record_time(self):    #定义一个方法 record_time，用于记录当前时间
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):    #定义一个方法 split_time，用于计算自上次记录以来的时间差
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time    #返回时间差

    def train(self, epoch, save_model=True):    #定义一个方法 train，接受轮数 epoch 和一个布尔参数 save_model，用于控制是否保存模型
        self.model.train()    #设置模型为训练模式
        self.print_log('Training epoch: {}'.format(epoch + 1))    #打印当前训练轮数
        loader = self.data_loader['train']    #获取训练数据加载器
        self.adjust_learning_rate(epoch)    #调整学习率

        loss_value = []    #初始化损失值和准确率的列表
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)    #将当前轮数记录到训练日志中
        self.record_time()    #记录当前时间
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)    #初始化时间记录字典，记录数据加载、模型前向传播和统计的时间
        process = tqdm(loader, dynamic_ncols=True)    #创建一个进度条，用于显示训练进度
        

        for batch_idx, (data, label, index) in enumerate(process):    #遍历训练数据，获取每个批次的数据、标签和索引
            self.global_step += 1    #更新全局步数计数器
            with torch.no_grad():    #在不计算梯度的上下文中执行以下操作
                data = data.float().cuda(self.output_device)    #将数据和标签转移到指定的 GPU 设备
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()    #记录数据加载时间

            # forward
            output = self.model(data)    #执行模型的前向传播，得到输出
            loss = sum([self.loss(out, label) for out in output])    #计算损失，针对每个输出进行损失计算并求和
            output = sum(output)    #将输出求和（可能是多头输出）

            # backward
            self.optimizer.zero_grad()    #清空优化器的梯度
            if self.arg.half:    #如果使用半精度训练
#                with torch.cuda.amp.autocast():  # 自动选择合适的类型
#                    outputs = self.model(inputs)  # 前向传播
#                    loss = self.criterion(outputs, labels)  # 计算损失

#                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:    #使用 Apex 进行半精度损失缩放和反向传播
                    loss.backward()
            else:    #否则，正常反向传播
                loss.backward()
            self.optimizer.step()    #更新优化器的参数
#            self.scaler.update()     # 更新缩放器

            loss_value.append(loss.data.item())    #将损失值添加到损失列表中
            timer['model'] += self.split_time()    #记录模型计算时间

            value, predict_label = torch.max(output.data, 1)    #获取每个样本的最大值和对应的预测标签
            acc = torch.mean((predict_label == label.data).float())    #计算当前批次的准确率
            acc_value.append(acc.data.item())    #将准确率添加到准确率列表中
            self.train_writer.add_scalar('acc', acc, self.global_step)    #将当前批次的准确率和损失记录到训练日志中
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']    #获取当前学习率
            self.train_writer.add_scalar('lr', self.lr, self.global_step)    #将当前学习率记录到训练日志中
            
            timer['statistics'] += self.split_time()    #记录统计时间
            process.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.data.item(), self.lr))    #更新进度条的描述，显示当前的损失和学习率

        # statistics of time consumption and loss
        proportion = {    #计算各个部分的时间占比
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(    #打印平均训练损失和平均准确率
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))    #打印各部分时间消耗的比例

        if save_model:    #如果需要保存模型
            state_dict = self.model.state_dict()    #获取模型的状态字典（参数）
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])    #将状态字典中的参数转移到 CPU，并去掉 'module.' 前缀

            torch.save(weights, self.savepath + '/epoch_' + str(epoch+1) + '_' + str(int(self.global_step)) + '.pt')    #保存模型的参数到指定路径

    #定义一个方法 eval，用于模型评估，接受参数包括评估轮数、是否保存分数、加载器名称、错误文件和结果文件
    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:    #如果提供了错误文件名，则打开该文件以写入错误预测的样本
            f_w = open(wrong_file, 'w')
        if result_file is not None:    #如果提供了结果文件名，则打开该文件以写入预测结果
            f_r = open(result_file, 'w')
        self.model.eval()    #将模型设置为评估模式
        self.print_log('Eval epoch: {}'.format(epoch + 1))    #打印当前评估轮数
        for ln in loader_name:    #遍历每个数据加载器名称，初始化损失值、分数片段、真实标签列表和预测标签列表
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0    #初始化步骤计数器和进度条，用于显示评估进度
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):    #遍历评估数据，获取每个批次的数据、标签和索引
                label_list.append(label)    #将当前批次的标签添加到标签列表中
                with torch.no_grad():    #在不计算梯度的上下文中执行以下操作
                    data = data.float().cuda(self.output_device)    #将数据和标签转移到指定的 GPU 设备
                    label = label.long().cuda(self.output_device)

                    output = self.model(data)    #执行模型的前向传播，得到输出
                    loss = sum([self.loss(out, label) for out in output])    #计算损失
                    output = sum(output)    #将输出求和

                    score_frag.append(output.data.cpu().numpy())    #将输出和损失值添加到对应的列表中
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)    #获取每个样本的最大值和对应的预测标签
                    pred_list.append(predict_label.data.cpu().numpy())    #将预测标签添加到预测列表，并增加步骤计数
                    step += 1

                if wrong_file is not None or result_file is not None:    #如果需要写入错误或结果文件
                    predict = list(predict_label.cpu().numpy())    #将预测和真实标签转换为列表
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):    #遍历预测列表
                        if result_file is not None:    #如果需要结果文件，将预测和真实标签写入文件
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:    #如果预测错误且需要错误文件，写入索引、预测和真实标签
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)    #将所有分数片段合并为一个数组
            loss = np.mean(loss_value)    #计算平均损失
            if 'ucla' in self.arg.feeder:    #如果数据集为 UCLA，更新样本名称
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)    #计算准确率，获取 top-1 准确率
            if accuracy > self.best_acc:    #如果当前准确率超过历史最佳，则更新最佳准确率和轮数
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)    #打印当前准确率和模型名称
            if self.arg.phase == 'train':    #如果当前阶段为训练
                self.val_writer.add_scalar('loss', loss, self.global_step)    #将损失和准确率记录到验证日志中
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(    #创建一个字典，将样本名称与分数对应起来
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(    #打印当前评估数据集的平均损失
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:    #打印 top-k 准确率
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:    #如果需要保存分数，则将分数字典保存到文件中
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.savepath, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)    #合并所有标签和预测列表
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)    #计算混淆矩阵
            list_diag = np.diag(confusion)    #获取混淆矩阵的对角线（正确预测）和每行的总和（真实样本数）
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum    #计算每个类别的准确率
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.savepath, epoch + 1, ln), 'w') as f:    #将每个类别的准确率和混淆矩阵写入 CSV 文件
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):    #定义一个方法 start，用于开始训练或测试过程
        if self.arg.phase == 'train':    #检查当前模式是否为训练模式
            self.print_log('Modelargs:\n{}\n'.format(str(vars(self.arg))))    #打印当前模型参数的详细信息
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size    #计算当前的全局步数，基于起始轮数、训练数据加载器的长度和批大小
            self.data_shape = [1, 3, self.arg.train_feeder_args['window_size'], self.arg.model_args['num_point'], 2]    #定义输入数据的形状，包括批大小、通道数、窗口大小、点数等
            flops, params = thop.profile(import_class(self.arg.model)(**self.arg.model_args), inputs=torch.rand([1] + self.data_shape), verbose=False)    #使用 thop 库计算模型的 FLOPs（每秒浮点运算次数）和参数数量，通过随机输入模拟模型
            self.print_log('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))    #打印模型的性能分析结果，包括 FLOPs 和参数数量
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):    #开始一个循环，从起始轮数遍历到总轮数
                self.print_log('*'*100)
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (    #判断是否需要保存模型，条件是当前轮数满足保存间隔或为最后一轮，并且超过了保存的起始轮数
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)    #调用 train 方法进行训练，传入当前轮数和是否保存模型的标志
                
                if epoch > self.arg.num_epoch-20:    #如果当前轮数大于总轮数的最后 20 轮
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])    #进行评估，使用测试数据加载器
                elif epoch%1==0:    #如果当前轮数是 1 的倍数
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])    #进行评估，使用测试数据加载器
                self.print_log('Best_Accuracy: {:.2f}%, epoch: {}'.format(self.best_acc*100, self.best_acc_epoch))    #打印当前的最佳准确率和对应的轮数

            # test the best model    以下代码用于测试最佳模型
            weights_path = glob.glob(self.savepath + '/epoch_' + str(self.best_acc_epoch) + '*')[0]    #使用 glob 查找最佳模型的权重文件
            
            weights = torch.load(weights_path)    #加载权重文件
            if type(self.arg.device) is list:    #检查设备参数是否为列表
                if len(self.arg.device) > 1:    #如果设备列表长度大于 1
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])    #将权重转移到指定的 CUDA 设备，并为每个权重添加模块前缀
            self.model.load_state_dict(weights)    #将加载的权重赋值给模型

            wf = weights_path.replace('.pt', '_wrong.txt')    #生成错误预测结果文件的路径
            rf = weights_path.replace('.pt', '_right.txt')    #生成正确预测结果文件的路径
            self.arg.print_log = False    #禁用打印日志功能
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)    #评估模型，保存得分，并将错误和正确的预测结果写入文件
            self.arg.print_log = True    #重新启用打印日志功能


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)    #计算模型中可训练参数的总数
            self.print_log('Best accuracy: {}'.format(self.best_acc))    #打印最佳准确率
            self.print_log('Epoch number: {}'.format(self.best_acc_epoch))    #打印最佳准确率对应的轮数
            self.print_log('Model name: {}'.format(self.arg.work_dir))    #打印模型名称或工作目录
            self.print_log('Model total number of params: {}'.format(num_params))    #打印模型的总参数数量
            self.print_log('Weight decay: {}'.format(self.arg.weight_decay))    #打印权重衰减参数
            self.print_log('Base LR: {}'.format(self.arg.base_lr))    #打印基础学习率
            self.print_log('Batch Size: {}'.format(self.arg.batch_size))    #打印批量大小
            self.print_log('Test Batch Size: {}'.format(self.arg.test_batch_size))    #打印测试阶段的批量大小
            self.print_log('seed: {}'.format(self.arg.seed))    #打印随机种子

        elif self.arg.phase == 'test':    #如果当前模式为测试模式
            wf = self.arg.weights.replace('.pt', '_wrong.txt')    #生成错误预测结果文件的路径
            rf = self.arg.weights.replace('.pt', '_right.txt')    #生成正确预测结果文件的路径

            if self.arg.weights is None:    #检查是否指定了权重文件
                raise ValueError('Please appoint --weights.')    #如果没有指定，抛出值错误
            self.arg.print_log = False    #禁用打印日志功能
            self.print_log('Model:   {}.'.format(self.arg.model))    #打印模型名称
            self.print_log('Weights: {}.'.format(self.arg.weights))    #打印权重文件路径
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)    #评估模型，保存得分，并将错误和正确的预测结果写入文件
            self.print_log('Done.\n')

def str2bool(v):    #定义一个函数 str2bool，用于将字符串转换为布尔值
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(import_str):    #定义一个函数 import_class，用于动态导入模块中的类
    mod_str, _sep, class_str = import_str.rpartition('.')    #使用 rpartition 将字符串按最后一个点分割为模块名和类名
    __import__(mod_str)    #动态导入模块
    try:
        return getattr(sys.modules[mod_str], class_str)    #从导入的模块中获取指定的类
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))    #抛出一个导入错误，包含类名和错误信息

if __name__ == '__main__':    #检查当前文件是否是主程序入口
    warnings.filterwarnings("ignore")    #忽略所有警告信息
    parser = get_parser()    #调用 get_parser 函数获取命令行参数解析器
    os.chdir(os.getcwd())    #将当前工作目录设置为当前目录

    p = parser.parse_args()    #解析命令行参数，并将结果存储在变量 p 中
    if p.config is not None:    #检查是否提供了配置文件参数
        with open(p.config, 'r') as f:    #如果提供了，打开指定的配置文件
#             default_arg = yaml.load(f)    #使用 yaml.load 读取 YAML 文件
            default_arg = yaml.load(f, Loader=yaml.FullLoader)    #使用 yaml.FullLoader 加载 YAML 文件内容，并将其存储在 default_arg 中
        key = vars(p).keys()    #获取命令行参数的所有键
        for k in default_arg.keys():    #遍历配置文件中的所有键
            if k not in key:    #检查配置文件中的键是否在命令行参数中
                print('WRONG ARG: {}'.format(k))
                assert (k in key)    #使用断言确保所有配置文件的键都在命令行参数中
        parser.set_defaults(**default_arg)    #将配置文件中的参数设置为命令行解析器的默认值

    arg = parser.parse_args()    #再次解析命令行参数，确保使用了更新后的默认值
    init_seed(arg.seed)    #调用 init_seed 函数，使用指定的随机种子初始化随机数生成器
    processor = Processor(arg)     #创建 Processor 类的实例，传入命令行参数
    processor.start()    #调用 Processor 实例的 start 方法，开始训练或测试过程
