
from pickle import FALSE
from config import Config
import fire
import numpy as np
import torch

# 随机数种子
seed = Config().seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class Autitude_Recognition(object):
    def __init__(self, config: Config=None):
        super().__init__()

        # 加载配置文件
        if config is None:
            config = Config()

        self.config = config

         # 指定训练设备
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(config.cuda_device_number))
            torch.backends.cudnn.benchmark = True
        
        print('device: {}'.format(self.device))

    def train_rnn(self):
        
        from model_pytorch.rnn_model import Trainer

        trainer = Trainer(self.config, self.device)
        # torch.backends.cudnn.benchmark = True
        trainer.train()
        trainer.test()

    def test_rnn(self):
        from model_pytorch.rnn_model import Trainer

        trainer = Trainer(self.config, self.device, is_train=False)
        # torch.backends.cudnn.benchmark = False
        trainer.test()

    def train_svm(self):
        from model_sklearn.ml_model import Trainer_SVM
        trainer = Trainer_SVM(self.config, is_train=True)
        trainer.train()

    def train_adaboost(self):
        from model_sklearn.ml_model import Train_Adaboost
        trainer = Train_Adaboost(self.config, is_train=True)
        trainer.train()
    
    def train_xgboost(self):
        from model_sklearn.ml_model import Train_XGboost
        trainer = Train_XGboost(self.config, is_train=True)
        trainer.train()

    def train_knn(self):
        from model_sklearn.ml_model import Train_KNN
        trainer = Train_KNN(self.config, is_train=True)
        trainer.train()


if __name__ == '__main__':
    
     # 设置默认为FloatTensor
    torch.set_default_tensor_type(torch.FloatTensor)

    # 解析命令行参数，执行指定函数
    fire.Fire(component=Autitude_Recognition())
    