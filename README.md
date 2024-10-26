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

### Data Processing

#### Generating Data

- Generate data.npz:

```
find following script in ./data/ and run:
data_prepare.py
```


# Training & Testing

### Training

- Change the parameters in config file or command line if needed: ./config/default.yaml

```
# Example: train
correct config file:test_feeder_args:data_path:data/data.npz

python main.py
```

```
# Example: test dataA(choose the best acc weights file)
correct config file:test_feeder_args:data_path:data/data.npz

python main.py --phase test --save-score True --weights ./work_dir/jbf_52/epoch_94_24064.pt

# Example: generate dataB pred
correct config file:test_feeder_args:data_path:data/B_data.npz

python main.py --phase test --save-score True --weights ./work_dir/jbf_52/epoch94_24064.pt

find the score file in ./work_dir/xxx/epoch1_test_score.pkl
turn the score into pred file:run turn.py(please correct the score file location)
```
