import sys
from tqdm import tqdm
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score,recall_score,classification_report
import joblib

sys.path.append('..')
sys.path.append('.')

from utils.function import *
from config import Config
from utils.data_process import *
from utils.logger import Logger

log = Logger('prepare_data').get_logger()

def flatten_pad(data: list, max_column: int = 1024):
    
    ret = []
    for d in data:
        ret.extend(d)
        if (len(ret)) >= max_column:
            break

     # 太长，截断
    if len(ret) > max_column:
        ret = ret[0: max_column]
    
    # 太短，填0
    if len(ret) < max_column:
        ret.extend([0.0 for _ in range(max_column - len(ret))])

    return ret


class Trainer_obj():
    def __init__(self,config: Config, is_train: bool=True):

        if is_train:
            self.train_x, self.train_y =  process_new_data(normalization=False)
            # self.train_x, self.eval_x, self.train_y, self.eval_y = train_test_split(data_x, data_y, 
                                        # test_size=config.eval_set_szie,random_state=config.seed)

        self.test_x, self.test_y = process_test_data(normalization=False)
        self.config = config

    def train(self):
        pass

    def test(self):
        pass


class Trainer_SVM(Trainer_obj):
    def __init__(self, config: Config, is_train: bool):
        super().__init__(config, is_train=is_train)

        self.model_file = './model_file/svm.pkl'
        self.test_x_1d = np.array([np.array(data_sample(data, sample_interval=30, max_column=1200)) for data in self.test_x])
    
    def train(self):
        train_x = np.array([np.array(data_sample(data, sample_interval=30, max_column=1200)) for data in self.train_x])
        
        # train_x = np.array([np.array(flatten_pad(data, max_column=2048)) for data in self.train_x])
        # test_x = np.array([np.array(flatten_pad(data, max_column=2048)) for data in self.eval_x])

        from sklearn.svm import LinearSVC

        liner_svc = LinearSVC(verbose=1, max_iter=10000)
        liner_svc.fit(train_x, self.train_y)

        joblib.dump(liner_svc, self.model_file)

        y_pred = liner_svc.predict(self.test_x_1d)
        # acc = np.sum(y_pred == self.test_y) / len(y_pred)

        print(classification_report(self.test_y, y_pred))

    def test(self):
        liner_svc = joblib.load(self.model_file)

        y_pred = liner_svc.predict(self.test_x_1d)
        print(classification_report(self.test_y, y_pred))



class Train_Adaboost(Trainer_obj):
    def __init__(self, config: Config, is_train: bool):
        super().__init__(config, is_train=is_train)

        self.model_file = './model_file/adaboost.pkl'
        self.test_x_1d = np.array([np.array(data_sample(data, sample_interval=10, max_column=768)) for data in self.test_x])
        # self.test_x_1d = np.array([np.array(flatten_pad(data, max_column=2048)) for data in self.test_x])


    def train(self):
        train_x = np.array([np.array(data_sample(data, sample_interval=10, max_column=768)) for data in self.train_x])
        # train_x = np.array([np.array(flatten_pad(data, max_column=2048)) for data in self.train_x])


        # adaboost = AdaBoostClassifier(n_estimators=50,random_state=0)

        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6, min_samples_leaf=6), algorithm='SAMME', n_estimators=50, learning_rate=0.5)
        adaboost.fit(train_x, self.train_y)


        y_pred = adaboost.predict(self.test_x_1d)
        
        joblib.dump(adaboost, self.model_file)

        print(classification_report(self.test_y, y_pred))

    def test(self):
        adaboost = joblib.load(self.model_file)

        y_pred = adaboost.predict(self.test_x_1d)
        print(classification_report(self.test_y, y_pred))

class Train_XGboost(Trainer_obj):
    def __init__(self, config: Config, is_train: bool):
        super().__init__(config, is_train=is_train)
        
        self.model_file = './model_file/xgboost.pkl'

        self.test_x_1d = np.array([np.array(data_sample(data, sample_interval=15, max_column=768)) for data in self.test_x])
        # self.test_x_1d = np.array([np.array(flatten_pad(data, max_column=2048)) for data in self.test_x])

    def train(self):
        train_x = np.array([np.array(data_sample(data, sample_interval=15, max_column=768)) for data in self.train_x])
        # train_x = np.array([np.array(flatten_pad(data, max_column=2048)) for data in self.train_x])


        from xgboost.sklearn import XGBClassifier
    
        xgboost = XGBClassifier(use_label_encoder=False,objective='multi:softprob')
        xgboost.fit(train_x, self.train_y)

        y_pred = xgboost.predict(self.test_x_1d)

        joblib.dump(xgboost, self.model_file)

        print(classification_report(self.test_y, y_pred))

    def test(self):
        xgboost = joblib.load(self.model_file)

        y_pred = xgboost.predict(self.test_x_1d)
        print(classification_report(self.test_y, y_pred))     

class Train_KNN(Trainer_obj):
    def __init__(self, config: Config, is_train: bool):
        super().__init__(config, is_train=is_train)
        
        self.model_file = './model_file/xgboost.pkl'

        self.test_x_1d = np.array([np.array(data_sample(data, sample_interval=15, max_column=768)) for data in self.test_x])
        # self.test_x_1d = np.array([np.array(flatten_pad(data, max_column=2048)) for data in self.test_x])

    def train(self):
        train_x = np.array([np.array(data_sample(data, sample_interval=15, max_column=768)) for data in self.train_x])
        # train_x = np.array([np.array(flatten_pad(data, max_column=2048)) for data in self.train_x])


        from sklearn.neighbors import KNeighborsClassifier
    
        knn = KNeighborsClassifier()
        knn.fit(train_x, self.train_y)

        y_pred = knn.predict(self.test_x_1d)

        joblib.dump(knn, self.model_file)

        print(classification_report(self.test_y, y_pred))

    def test(self):
        knn = joblib.load(self.model_file)

        y_pred = knn.predict(self.test_x_1d)
        print(classification_report(self.test_y, y_pred))     

if __name__ == '__main__':
    a = [
        [1,2,3,4],
        [1,1,1,1],
        [2,2,3,3],
    ]

    ret = data_sample(a, sample_interval=6, max_column=32)

    print(len(ret))
    print(ret)
    