import os, sys
import codecs, csv, ujson
from fastapi.param_functions import File
from numpy.lib.function_base import average
from tqdm import tqdm
import numpy as np
from scipy.signal import savgol_filter

sys.path.append(".")
sys.path.append("..")
from utils.logger import Logger
from utils.function import *

log = Logger('prepare_data').get_logger()

# 采集数据保存的路径
DATA_PATH = "./data/new_data2/"
SENSOR_TYPE_JSON =  "./sensor_type.json"
TEST_DIR = './data/test_data2/'
SENSOR_TYPE = None
# kf = Kalman_Filter()
def init_sensortype():
 # 加载传感器类型json文件，用于将传感器类型的文本转换为id
    global SENSOR_TYPE
    with codecs.open(SENSOR_TYPE_JSON) as f:
        SENSOR_TYPE = ujson.load(f)

init_sensortype()

def process_old_version_data(data_path: str=DATA_PATH, sensor_type_file: str=SENSOR_TYPE_JSON, save: bool = False):
    '''
    处理老版本的数据，将老版本的数据转换为新版本数据
    '''
    sensor_type = SENSOR_TYPE

    # 获取data目录下的所有文件
    file_names = []
    for file_path, dirname, files in os.walk(data_path):
       file_names = files

    all_data_x = []
    all_data_y = []

    log.info("preocess data...")

    def getOneSampleData(file: File):
        current_sample_x = []
        reader = csv.reader(file)
        current_time = None
        current_data_x = [0.0 for _ in range(len(sensor_type) * 3)]

        for row in reader:
            if current_time is None:
                current_time = row[0]

            if current_time != row[0]:
                
                current_data_x.insert(0,current_time)
                current_time = row[0]
                current_sample_x.append(current_data_x)
                current_data_x = [0.0 for _ in range(len(sensor_type) * 3)]


            # xyz三个轴的数据
            for i, value in enumerate(row[2: ]):
                current_data_x[sensor_type[row[1]] * 3 + i] = eval(value)
        # current_sample_x = std_normalzation(current_sample_x)
        return current_sample_x

    for file in tqdm(file_names):
       
        current_data_y = int(file.split('.')[0])
        current_sample_x = []
        with codecs.open(data_path + file, 'r', encoding='utf-8') as f:
            current_sample_x = getOneSampleData(f)
            with codecs.open('./data/processed/' + file, 'w', encoding='utf-8') as ff:
                writer = csv.writer(ff)
                writer.writerow(['timestamp','gravity_x','gravity_y','gravity_z','linear_accel_x','linear_accel_y','linear_accel_z','accel_x','accel_y',
                'accel_z','magnet_x','magnet_y','magnet_z','gyro_x','gyro_y','gyro_z'])
                writer.writerows(current_sample_x)

        all_data_x.append(current_sample_x)
        all_data_y.append(current_data_y)      

    # end for
    
    # if save:
    #     with codecs.open("./data/data_x.csv", 'w', encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerows(all_data_x)

    #     with codecs.open("./data/data_y.csv", 'w', encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerows(all_data_y)


def process_new_data(data_path: str=DATA_PATH, normalization: bool=True):
    '''
    将新版本数据加载到内存
    '''
    # 获取data目录下的所有文件
    file_names = []
    for file_path, dirname, files in os.walk(data_path):
       file_names = files

    all_data_x = []
    all_data_y = []

    class_count = {
        0:0,
        1:0,
        2:0,
    }

    log.info("preocess new data...")

    for file in tqdm(file_names):
           
        current_data_y = int(file.split('.')[0])
        class_count[current_data_y] += 1;

        current_sample_x = []
        with codecs.open(data_path + file, 'r', encoding='utf-8') as f:
            f.readline() # 去除表头
            current_sample_x = process_post_file(f,start=1, normalization=normalization)

        all_data_x.append(np.array(current_sample_x))
        all_data_y.append(current_data_y)

    log.info(class_count)
    return all_data_x, all_data_y

def process_post_file(str_list, start: int = 0, normalization: bool=False):
    '''
    '''
    sensor_type = SENSOR_TYPE
    current_sample_x = []
    reader = csv.reader(str_list)
    
    for i, row in enumerate(reader):
        current_data_x = [0.0 for _ in range(len(sensor_type) * 3)]

        # 检查是否为15列
        if len(row) - start != 15:
            raise Exception("data column is not equal 15, in line {}".format(i + 1))

        # xyz三个轴的数据
        for i, value in enumerate(row[start: ]):
            current_data_x[i] = eval(value)

        # 去掉磁场
        current_data_x = current_data_x[0: 9] + current_data_x[12: ]
        # print(current_data_x)
        # exit()

        # 去掉 linear acc
        # current_data_x = current_data_x[0: 3] + current_data_x[6:]
        
        current_sample_x.append(current_data_x)

    if normalization:
        current_sample_x = max_min_normalization(current_sample_x)
    # current_sample_x = std_normalzation(current_sample_x)
    # current_sample_x = kf.do_smooth(np.array(current_sample_x))
    # current_sample_x = savgol_filter(current_sample_x, 3, 2)
    
    return current_sample_x


def process_test_data(dir: str= TEST_DIR, normalization: bool=True):

    file_names = []
    for file_path, dirname, files in os.walk(dir):
           file_names = files

    all_data_x = []
    all_data_y = []

    log.info("preocess test data...")

    for file in tqdm(file_names):
       
        current_data_y = int(file.split('.')[0])
        current_sample_x = []
        with codecs.open(dir + file, 'r', encoding='utf-8') as f:
            f.readline()
            current_sample_x = process_post_file(f,start=1, normalization=normalization)
           
        all_data_x.append(np.array(current_sample_x))
        all_data_y.append(current_data_y)      
    
    return all_data_x, all_data_y

if __name__ == '__main__':
    # process_test_data()
    # x, y = process_new_data()
    # print(x[0],x[0].shape)
    # exit()
    
    import numpy as np

    a = []
    for i in range(3):
        a.append(np.array([j+i for j in range(4)]))
    a = np.array(a)
    print(a)

    # mean = np.mean(a,axis=0)
    # sigma = np.std(a, axis=0)
    # print(sigma)
    # print(mean)
    # b = (a - mean) / sigma




    # print(b)

    