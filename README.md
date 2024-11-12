# 环境配置

```
- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX, sklearn, matplotlib, thop
```

# 数据准备

## DEGCN_pro工程下：

```
- data/
 - train_joint.npy
 - train_label.npy
 - test_joint.npy
 - test_label.npy
 - val_joint.npy
```

# 数据预处理

## DEGCN_pro工程下：

在`./data`文件夹找到以下数据预处理文件并按顺序运行：

```
data_rid_0.py
data_norm.py
generate_TEdata.py
```

## TEGCN工程下：

在`./DEGCN/data`文件夹找到以下文件夹并移动到`./TEGCN/data`下，移动时把文件夹里的数据直接移到`./TEGCN/data`下：

```
- data/
 - TEgcn_有数据预处理
 - TEgcn_无数据预处理
```

生成符合tegcn要求的标签文件和生成其他模态文件，见：

```
- TEGCN
  - data
    - transform.py
```

```
- TEGCN
  - gen_modal.py
```

# 复现过程

得到最终结果用到的10个权重以及日志，评估集测试集置信度文件位置:

```
 -DEGCN_pro
     - work_dir
       - result0
         - weight.pt
         - log.txt
         - score.pkl
         - test
           - test_score.pkl
       - result1
       - result2
       - result3
       - result4
       - result5
       - result6
       - result7
       - result8
       - result9  
```

如何融合各权重置信度见：

```
- DEGCN_pro
  - emerge_score.py
```

生成的测试集权重位置：

```
- DEGCN_pro
  - work_dir
    - emerge
      - pred.npy
```

### 训练得到权重过程：

根据每个结果的日志内容训练出权重：

​       1.选择模型：按照日志模型内容选择在degcn_pro或tegcn下复现训练权重

​       2.修改参数：

​                由于数据预处理细节偏差和模型本身数据增强，不可能复现出同样的权重，但是可以将训练日志            的参数（模型、正则化参数、数据路径）改到参数配置文件中：`DEGCN_pro\config\default.yaml`与`TEGCN\config\train.yaml`；

​       3.直接运行两个工程下的`main.py`

### 测试相应权重过程：

 #### 用权重测试评估集：

#### DEGCN_pro:

​         修改参数：

​                 参数配置文件`.\config\default.yaml`中修改数据路径，改为评估集：`test_feeder_args:data_path: data/val_data.npz`；

​         命令行输入：

​                 按照要测试的权重运行以下示例命令（需要修改模型以及权重路径）:

```
python main.py --phase test --save-score True --weights ./work_dir/76_6_weights/76_6.pt --model model.degcn.Model
```

#### TEGCN:

​         修改参数：

​                  参数配置文件`.\config\test.yaml`中修改数据路径，改为评估集`test_feeder_args:  data_path: ./data/new_vel_bone.npy`（示例，实际情况按照日志内容改）

​                  修改TEGCN\scripts\EVAL_V1.sh的权重路径，运行该文件。

#### 用权重测试测试集:

​        步骤同上，区别是修改的数据路径不同，改为测试集数据

#### 融合测试结果得到最终置信度过程：

​        如何融合各权重置信度见：

```
- DEGCN_pro
  - emerge_score.py
```

​        在代码中输入置信度文件路径，按照给定权重（代码中已给出）融合置信度文件;

​        注意选择融合评估集还是测试集
