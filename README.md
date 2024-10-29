
# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX, sklearn, matplotlib, thop

# Data Preparation

- data
 - train_joint.npy
 - train_label.npy
 - test_A_joint.npy
 - test_A_label.npy
 - test_joint_B.npy

#自己按照路径和命名格式导入赛方提供的原始数据

### Data Processing

- Generate data.npz:

find following script in ./data/ and run:
#在./data文件夹找到以下数据预处理文件并按顺序运行：
data_rid_0.py
data_denoise.py
data_denoise_3.py


# Training & Testing

- Change the parameters in config file or command line if needed: ./config/default.yaml
#可在./config/default.yaml修改训练或测试参数

# Example: train
correct config file:test_feeder_args:data_path:data/data.npz #训练时指定评估集数据为A
python main.py


# Example: test dataA(choose the best acc weights file) #选择相应权重测试数据集A
correct config file:test_feeder_args:data_path:data/data.npz #测试集数据指定为A
python main.py --phase test --save-score True --weights ./work_dir/jbf_52/epoch_94_24064.pt
#命令行输入指定参数运行


# Example: generate dataB pred #选择相应权重生成B的置信度文件
correct config file:test_feeder_args:data_path:data/B_data.npz #测试集数据指定为B
python main.py --phase test --save-score True --weights ./work_dir/jbf_52/epoch94_24064.pt
在./work_dir/test/时间戳文件夹找到生成的置信度文件pkl
运行turn.py转化成数组文件（自己修改文件路径即可）


# Example: 加权融合几个置信度文件生成综合置信度（选择相应pkl置信度文件进行融合）
emerge_score.py
自行修改pkl文件路径（用相应权重和参数跑出的结果）
自行修改融合权重，保证权重之和为1


# 结果复现
比赛得到最好结果的复现过程（训练过程具有随机参数，不可能100%训练出一模一样的权重）：
   一 用于融合的两权重和参数路径：
      1.
      2.
   二 用相同参数训练模型得到权重：
      1.读取上述log文件开头的参数（主要是model、weight_decay、cosine_epoch、batch_size、num_epoch、
  test_feeder_args:data_path，修改参数到./config/default.yaml）
      2.运行main.py
      3.在./work_dir/train/时间戳文件夹获取最优权重
   二 测试数据集得到置信度pkl文件：
      根据自身需要选择数据集A或是B:
         correct config file:test_feeder_args:data_path:data/data.npz #测试集数据指定为A
         correct config file:test_feeder_args:data_path:data/B_data.npz #测试集数据指定为B
      命令行输入：
         python main.py --phase test --save-score True --weights ./work_dir/jbf_weights/epoch_94_24064.pt --model model.jbf.Model
         python main.py --phase test --save-score True --weights ./work_dir/jbf_weights/epoch_94_24064.pt --model model.jbf.Model
      找到pkl文件：
         默认在./work_dir/test/时间戳文件夹里，已经测试出的置信度路径：
            测试集A:
            测试集B:
   三 将pkl文件融合得到综合准确率或融合置信度数组：
      1.修改Pkl文件路径
      2.自定义权重（默认：
      3.根据需要选择代码块，注释其中一块
      4.获得结果
