from dataclasses import dataclass

@dataclass
class Config(object):
    
    # 随机数种子
    seed = 233333
    
    # 训练轮次
    epoch = 50

    batch_size = 32

    # 特征数量， 4 * 3
    fearure_size = 12

    learning_rate = 0.01

    rnn_hidden_size = 128
    rnn_layers = 2

    # 评估集占比
    eval_set_szie = 0.2

    # cuda显卡设置，默认0，其他数值在多显卡情况下生效
    cuda_device_number = 0

