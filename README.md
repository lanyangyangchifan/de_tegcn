```
# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX, sklearn, matplotlib, thop

# Data Preparation

- train_joint.npy
- train_label.npy
- test_A_joint.npy
- test_A_label.npy
- test_joint_B.npy

Location: ./data/

please import dataset by yourself
#自己按照路径和命名格式导入赛方提供的原始数据

### Data Processing

#### Generating Data

- Generate data.npz:

find following script in ./data/ and run:
#在./data文件夹找到以下数据预处理文件并按顺序运行：
data_rid_0.py
data_denoise.py
data_denoise_3.py


# Training & Testing

### Training

- Change the parameters in config file or command line if needed: ./config/default.yaml
#可在./config/default.yaml修改训练或测试参数

# Example: train
correct config file:test_feeder_args:data_path:data/data.npz #训练时指定测试集数据为A
python main.py


# Example: test dataA(choose the best acc weights file) #选择相应权重进行测试
correct config file:test_feeder_args:data_path:data/data.npz #测试集数据指定为A
python main.py --phase test --save-score True --weights ./work_dir/jbf_52/epoch_94_24064.pt
#命令行输入指定参数运行

# Example: generate dataB pred #选择相应权重生成B的置信度文件
correct config file:test_feeder_args:data_path:data/B_data.npz #测试集数据指定为B

python main.py --phase test --save-score True --weights ./work_dir/jbf_52/epoch94_24064.pt

在./work_dir/test/时间戳文件夹找到生成的置信度文件pkl
运行turn.py转化成数组文件（自己修改文件路径即可）
```
