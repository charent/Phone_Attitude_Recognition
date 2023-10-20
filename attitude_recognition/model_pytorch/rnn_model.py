import sys
from os.path import dirname, abspath

from sklearn.metrics import accuracy_score,recall_score,classification_report
from sklearn.model_selection  import train_test_split

from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

sys.path.append('..')
sys.path.append('.')

from utils.function import *
from config import Config
from utils.data_process import *


parent_path = abspath(dirname(dirname(__file__)))

log = Logger('LSTM_Model').get_logger()

# 三分类
OUTPUT_CLASS = 3

class SensorDataSet(Dataset):
    def __init__(self, data_x: list, data_y: list) -> None:
        super().__init__()

        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)


   
def collate_fn(data):
    '''
    将list or ndarray转换为tensor，并且对齐数据
    '''
    data_x = [item[0] for item in data]
    data_y = [item[1] for item in data]
    
     # 对齐数据
    data_x, lengths = array_pad(data_x, return_length=True)
    
    as_tensor = torch.as_tensor

    ret = {
        'train_x': as_tensor(data_x, dtype=torch.float32),
        'train_y': as_tensor(data_y, dtype=torch.long),
        'lengths': as_tensor(lengths, dtype=torch.long)
    }

    return ret


class RNN_Model(nn.Module):
    def __init__(self, feature_size: int, rnn_layers: int = 2, hidden_size: int = 64, output_class: int=OUTPUT_CLASS):
        super(RNN_Model, self).__init__()

        # self.layernorm = nn.LayerNorm((feature_size))

        self.rnn = nn.GRU(
            input_size=feature_size, 
            hidden_size=hidden_size, 
            num_layers=rnn_layers, 
            batch_first=True, 
            bidirectional=False,
        )

        self.droupout = nn.Dropout(0.2)
        

        self.out_fc = nn.Sequential(
            nn.Linear( 
                    in_features=hidden_size * 2, 
                    out_features=output_class,
                ),
        )


    def forward(self, inputs: Tensor, lengths: Tensor):
        '''
        inputs:
            (batch_size, seq_len, feature_size)
        lengths:
            (batch_size, )
        '''
        device = inputs.device
        # inputs = self.layernorm(inputs)

        # 
        max_seq_len = inputs.shape[1]

        inputs = pack_padded_sequence(input=inputs, lengths=lengths, batch_first=True, enforce_sorted=False)
        inputs, h_x = self.rnn(inputs)
        outs, _ = pad_packed_sequence(inputs, batch_first=True, total_length=max_seq_len)

        outs = self.droupout(outs)

        # 屏蔽掉填充部分，再去做池化
        mask = create_mask_from_lengths(lengths).to(device)
        max_pool = tensor_max_poll(outs, mask)
        avg_pool = tensor_avg_poll(outs, mask)

        # (batch_size, rnn_hidden_size * 4)
        inputs = torch.cat([max_pool, avg_pool], dim=1)

        # (batch_size, 1)
        outs = self.out_fc(inputs)
    
        return outs
        


class Trainer():
    def __init__(self, config: Config, device: str, is_train: bool=True):

        if is_train:
            data_x, data_y =  process_new_data(normalization=True)
            self.train_x, self.eval_x, self.train_y, self.eval_y = train_test_split(data_x, data_y, 
                                        test_size=config.eval_set_szie,random_state=config.seed)

        self.test_x, self.test_y = process_test_data(normalization=True)
  
        self.config = config
        self.device = device

    def test(self):
        config = self.config
        device = self.device

        test_data_loader = DataLoader(
            dataset=SensorDataSet(
                data_x=self.test_x,
                data_y=self.test_y,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )

        model_save_path = parent_path +'/model_file'

        lstm_model = RNN_Model(
            feature_size=config.fearure_size,
            rnn_layers=config.rnn_layers,
            hidden_size=config.rnn_hidden_size
        ).to(device)

        lstm_model.load_state_dict(torch.load("{}/gru.pkl".format(model_save_path), map_location=torch.device(device)))
        lstm_model.eval()

        
        with torch.no_grad():
            acc, recall, all_true_y, all_pred_y = self.evaluate(lstm_model, test_data_loader)
            info = 'test acc: {:.4f}, test recall: {:.4f}'.format(acc, recall)
            log.info(info)

            class_name = ['正常输入', '输入中换人', '行走输入']
            print()
            print(classification_report(all_true_y, all_pred_y))

    def train(self):
        config = self.config
        device = self.device
        train_data_loader = DataLoader(
            dataset=SensorDataSet(
                data_x=self.train_x,
                data_y=self.train_y,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )

        eval_data_loader = DataLoader(
            dataset=SensorDataSet(
                data_x=self.eval_x,
                data_y=self.eval_y,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )

        model_save_path = parent_path +'/model_file'

        lstm_model = RNN_Model(
            feature_size=config.fearure_size,
            rnn_layers=config.rnn_layers,
            hidden_size=config.rnn_hidden_size
        ).to(device)

    
        # 二分类交叉熵
        # loss_function = nn.BCEWithLogitsLoss().to(device)

        # 多分类交叉熵
        loss_function = nn.CrossEntropyLoss().to(device)

        # 网络参数
        params = get_models_parameters(model_list=[lstm_model])

        # 优化器
        optimizer = torch.optim.Adam(params=params, lr=config.learning_rate)

        steps = int(np.round(len(self.train_x) / config.batch_size))


        best_acc = 0.0
        best_epoch = 0
        loss_cpu = 0.0
        loss_sum = 0.0

        for epoch in range(config.epoch):
            device = self.device
            lstm_model.train()

            log.info('epoch: {}, average batch loss: {:.3f}'.format(epoch, loss_sum / steps))
            loss_sum = 0.0

            with tqdm(total=steps) as pbar:
                for step, inputs_outputs in enumerate(train_data_loader):
                    pbar.update(1)
                    pbar.set_description('epoch: {}'.format(epoch))
                    pbar.set_postfix_str('loss: {:0.3f}'.format(loss_cpu))

                    train_x = inputs_outputs['train_x'].to(device)
                    train_y = inputs_outputs['train_y'].to(device)
                    lengths = inputs_outputs['lengths'].to('cpu')   #长度只能在cpu上

                    pred_y = lstm_model(train_x, lengths)

                    # print()
                    # print(train_y.shape)
                    # print(pred_y.shape)

                    loss = loss_function(input=pred_y, target=train_y)
                 
                  
                    loss_cpu = loss.cpu().detach().numpy()
                    loss_sum += loss_cpu

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # end process bar

            lstm_model.eval()
            with torch.no_grad():
                acc, recall, all_true_y, all_pred_y = self.evaluate(lstm_model, eval_data_loader)
                if acc >= best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    torch.save(lstm_model.state_dict(), '{}/gru.pkl'.format(model_save_path))
                info = 'epoch: {}, loss: {:.4f}, best acc: {:.4f}, best epoch: {}, current acc: {:.4f}'.format(
                                        epoch, loss_sum / steps, best_acc, best_epoch, acc)
                log.info(info)

    
    def evaluate(self, model: nn.Module, eval_data_loader: DataLoader):
        device = self.device

        correct_count = 0

        all_true_y = []
        all_pred_y = []

        with tqdm() as pbar:
            for step, inputs_outputs in enumerate(eval_data_loader):
                pbar.update(1)
                pbar.set_description('eval: ')

                eval_x = inputs_outputs['train_x'].to(device)
                eval_y = inputs_outputs['train_y'].to(device)
                lengths = inputs_outputs['lengths'].to('cpu')   #长度只能在cpu上

                pred_y = model(eval_x, lengths)
                pred_y = torch.argmax(pred_y, dim=-1)

                correct_count += torch.sum(pred_y == eval_y).detach().cpu().numpy()
                all_true_y.extend(eval_y.detach().cpu().numpy().tolist())
                all_pred_y.extend(pred_y.detach().cpu().numpy().tolist())
    
    
        acc = accuracy_score(all_true_y, all_pred_y, )
        recall = recall_score(all_true_y, all_pred_y, average='macro')
  
 
        return acc, recall, all_true_y, all_pred_y


# if __name__ == '__main__':

#      # 设置默认为FloatTensor
#     torch.set_default_tensor_type(torch.FloatTensor)

    
#     config = Config()
#     device = 'cpu'
#     trainer = Trainer(config=config, device=device)
#     trainer.train(config=config, device=device)