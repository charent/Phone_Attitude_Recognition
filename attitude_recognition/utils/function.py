import re
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import ModuleList, Module
import torch.nn.functional as F
import codecs
import ujson
import numpy as np

def read_json(path: str):
    '''
    加载json文件
    '''
    with codecs.open(path, 'r', encoding='utf-8') as f:
        return ujson.load(f)

def create_mask_from_lengths(lengths: Tensor, max_len: int=None):
    '''
    通过lengths数组创建mask
    '''
    if max_len is None:
        max_len = torch.max(lengths)
    device = lengths.device
    mask = torch.arange(max_len).expand(len(lengths), max_len).to(device) < lengths.unsqueeze(1)
    mask = mask.float().detach()
    return mask

def f1_p_r_compute():
    '''
    计算spo的f1分数，精确率，召回率，
    '''
    pass

def tensor_max_poll(seq: Tensor, mask: Tensor):
    '''
    对seq进行mask，然后做max pool
    seq:  [batch_size, seq_len, embed_dim]
    mask: [bath_size, seq_len]
    '''
    if len(mask.shape) == 2:
        mask = torch.unsqueeze(mask, dim=2)
    seq = seq - (1 - mask) * 1e9

    return torch.max(seq, dim=1)[0]

def tensor_avg_poll(seq: Tensor, mask: Tensor):
    '''
    对seq进行mask，然后做average pool
    seq:  [batch_size, seq_len, embed_dim]
    mask: [bath_size, seq_len]
    '''
    if len(mask.shape) == 2:
        mask = torch.unsqueeze(mask, dim=2)
    seq = seq * mask
    length = torch.sum(mask, dim=1)
    
    return torch.sum(seq, dim=1) / length


def get_models_parameters(model_list: list, weight_decay: float=0.0001):
    '''
    获取多个模型的可训练参数，包括模型内嵌套的网络参数
    多个模型放到一个list里面
    '''
    parameters = []
    no_decay = ["bias", "LayerNorm.weight"]

    for model in model_list:
        params = [{
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },{
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        parameters.extend(params)
    
    return parameters

def init_weights(m: Module):
    '''
    参数权重初始化,
    使用方法：
        model = Net()
        model.apply(init_weights)
    '''
    # if isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight.data)
    #     if m.bias is not None:
    #         m.bias.data.fill_(0.0)
    # if isinstance(m, nn.Conv1d):
    #     nn.init.xavier_normal_(m.weight.data,)
    #     if m.bias is not None:
    #         m.bias.data.fill_(0.0)
    if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                nn.init.xavier_uniform_(p.data)
            elif 'weight_hh' in n:
                nn.init.orthogonal_(p.data)
            elif 'bias' in n:
                p.data.fill_(0.0)

def tensor_avg_pool1d(tensor: Tensor, kernel_size: int, mask: Tensor=None, stride: int=1):
    '''
    先对tensor做mask，再做1d averger poo，
    maske: [batch_size, seq_len]
    tensor: [batch_size, seq_len, embedding_dim]
    return: [batch_szie, seq_len, embedding_dim]
    '''
    if mask is not None:
        mask = mask.unsqueeze(2)
        tensor = tensor * mask
    ret = F.avg_pool1d(tensor, kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) / 2))

    return ret
    
def tensor_padding(t: Tensor, max_seq_len: int=256):
    '''
    对tensor进行填充，填充为最大长度：max_seq_len
    如果t的seq_len大于max_seq_len，则截断为max_seq_len
    t: 
        [batch_size, seq_len, embedding_dim]
    return:
        [batch_size, max_seq_len, embedding_dim]
    '''
    seq_len = t.shape[1]
    device = t.device

    if seq_len > max_seq_len:
        return t[:, 0: max_seq_len, :]

    batch_size = t.shape[0]
    embedding_dim = t.shape[2]
    pad_len = max_seq_len - seq_len

    pad_tensor = torch.zeros((batch_size, pad_len, embedding_dim)).to(device)
    ret = torch.cat([t, pad_tensor], dim=1)

    return ret

def array_pad(array: list, padding_value: float=0.0, return_length: bool=False):
    '''
    对不同长度的数组进行对齐，对齐长度为数组中的最大长度
    输入：
        array: list or ndarray, (batch_size, seq_len, feature_size)
    输出：
        array: (batch_size, max_seq_len, feature_size)
    '''
    ret = []
    append = ret.append

    # feature_size = array[0].shape[-1]

    lengths = [len(item) for item in array]

    max_seq_len = max(lengths)
  
    for item in array:
        # 矩阵的上下、左右填充的行数((0, 下面要填充的行数), (0，0))
        pad_edge = ((0, max_seq_len - len(item)), (0,0))
        item = np.pad(item, pad_edge, constant_values=(0.0, padding_value))
        append(item)
       
    ret = np.array(ret)

    if return_length:
        return ret, lengths
    
    return ret


def max_min_normalization(array):
    '''
    在列上的归一化，最大最小归一化

    '''
    min_ = np.min(array, axis=0)
    max_ = np.max(array, axis=0)
    
    ret = (array - min_) / (max_ - min_ + 1e-10)

    return ret


def std_normalzation(array):
    '''
    在列上做标准化
    x = (x - μ) / σ
    '''
    mean = np.mean(array,axis=0)
    sigma = np.std(array, axis=0)

    ret = (array - mean) / sigma

    return ret

def data_sample(data: list, sample_interval: int, max_column: int ):
    '''
    data = [[...],[...]]
    对data进行采样，sample_interval为采样间隔
    sample_interval>5 的时候才能起到压缩数据的作用
    采样后将结果拼接为一行，max_column控制这一行的最大列数（长度）
    一个data返回一行：[...]
    '''

    # 要迭代的总次数， 向上取整
    n = int(np.ceil(len(data) / sample_interval))

    ret = []
    
    for i in range(n):
        start = i * sample_interval
        end = start + sample_interval

        temp_result = []
        sample_list = data[start: end]

        # 求均值
        temp_result.extend(np.mean(sample_list, axis=0))
        
        # 求方差
        temp_result.extend(np.var(sample_list, axis=0))

        # 求标准差
        temp_result.extend(np.std(sample_list, axis=0))

        # 求最大值
        temp_result.extend(np.max(sample_list, axis=0))

        # 求最小值
        temp_result.extend(np.min(sample_list, axis=0))

        ret.extend(temp_result)
        
        if len(ret) >= max_column:
            break
    
    # 太长，截断
    if len(ret) > max_column:
        ret = ret[0: max_column]
    
    # 太短，填0
    if len(ret) < max_column:
        ret.extend([0.0 for _ in range(max_column - len(ret))])

    return ret

if __name__ == '__main__':
    a = [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ]

    a = np.random.random((15,12))
    print(a.shape)
    b = data_sample(a, 15, 65)
    print(b)
    print(len(b))

    exit(0)

    b = std_normalzation(a)
    print(b)
    