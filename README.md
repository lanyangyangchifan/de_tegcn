# 环境配置

```
- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX, sklearn, matplotlib, thop
```

# 数据准备

按照路径和命名格式导入赛方提供的原始数据：

```
- data/
 - train_joint.npy
 - train_label.npy
 - test_A_joint.npy
 - test_A_label.npy
 - test_joint_B.npy
```

# 数据预处理

在`./data`文件夹找到以下数据预处理文件并按顺序运行：

```
data_rid_0.py
data_correct.py
```

此外，`./data`文件夹下还有其他数据预处理文件会在后续操作用到

# 复现过程

得到最终结果用到的三个权重:

```
 - work_dir\74_95_weights\epoch_70_17920.pt
 - work_dir\75_85_weights\epoch_94_24064.pt
 - work_dir\76_6_weights\76_6.pt
```

与相应权重配套的训练日志:

```
 - work_dir\74_95_weights\log(74.95).txt
 - work_dir\75_85_weights\log(75.85).txt
 - work_dir\76_6_weights\log(76.6).txt
```

### 训练得到权重过程：

不足之处：

​        75_85_weights和76_6_weights训练时的数据预处理代码细节已丢失，测试时用的数据集是改进后的数据预处理代码得到的，因此得到的test_A的准确率与训练日志中的准确率有0.1%以内的出入。

训练出权重：

​        1.修改参数：

​                由于数据预处理细节偏差和模型本身数据增强，不可能复现出同样的权重，但是可以将训练日志            的参数改到参数配置文件中：`.\work_dirconfig\default.yaml`（麻烦老师按照日志内容手动改一下）；

​       2.数据预处理修改：

​               74_95_weights和76_6_weights在训练前应运行`data_norm_2.py`，75_85_weights在训练前应运行`data_norm_1.py`；

​       3.直接运行`main.py`

### 测试相应权重过程：

 #### 用权重测试数据集A：

​         数据预处理修改：

​                 74_95_weights和76_6_weights在测试前应运行`data_norm_2.py`，75_85_weights在测试前应运行`data_norm_1.py`；

​         修改参数：

​                 参数配置文件`.\work_dirconfig\default.yaml`中修改`test_feeder_args:data_path: data/data.npz`；

​         命令行输入：

​                 按照要测试的权重运行以下命令之一:

```
1.python main.py --phase test --save-score True --weights ./work_dir/76_6_weights/76_6.pt --model model.degcn.Model
2.python main.py --phase test --save-score True --weights ./work_dir/75_85_weights/epoch_94_24064.pt --model model.jbf.Model
3.python main.py --phase test --save-score True --weights ./work_dir/74_95_weights/epoch_70_17920.pt --model model.jbf.Model
```

#### 用权重测试数据集B:

​        步骤同上，区别是修改参数部分为`test_feeder_args:data_path: data/B_data.npz`

#### 融合测试结果得到最终置信度过程：

​        找到置信度文件：依照时间顺序`score.pkl`文件会出现在`'./work_dir/test/时间戳'`文件夹，自行匹配；
​        已有置信度文件：按照上述步骤跑出的置信度文件经过整理后分别在以下路径：

```
work_dir\test\test_A\74_95_test_score.pkl
work_dir\test\test_A\75_85_test_score.pkl
work_dir\test\test_A\76_65_test_score.pkl
work_dir\test\test_B\74_95_test_score.pkl
work_dir\test\test_B\75_85_test_score.pkl
work_dir\test\test_B\76_65_test_score.pkl 
```

​        找到融合功能代码：`.\emerge_score.py`

​        在代码中输入置信度文件路径，按照给定权重（代码中已给出`3：4：4`）融合置信度文件;

​        按照需要注释代码块：代码中已给出解释
