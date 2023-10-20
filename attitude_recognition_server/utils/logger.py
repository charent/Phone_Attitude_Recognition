import logging
from os.path import dirname, abspath
import os
import colorlog 

parent_path = abspath(dirname(dirname(__file__)))

class Logger(object):
    def __init__(self,logger_name: str, level=logging.DEBUG, std_out: bool=True, save2file: bool=False):
        super().__init__()

        if std_out == False and save2file == False:
            raise ValueError('args: [std_out, save2file], at less one of them must be True')

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        # 默认的格式化
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
        
        # 输出到控制台
        if std_out:
             # 彩色输出格式化
            log_colors_config = {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red'
            }
            formatter = colorlog.ColoredFormatter(
                        '[%(asctime)s] [%(levelname)s]： %(log_color)s%(message)s',
                        log_colors=log_colors_config
                        )
            ch = logging.StreamHandler()
            ch.setLevel(level)        
            ch.setFormatter(formatter)
            
            self.logger.addHandler(ch)
       
                    
         # 输出到文件
        if save2file:
            base_dir = parent_path + '/log' # 获取上级目录的绝对路径
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)
                
            log_file = base_dir + '/' + logger_name +'.log'
            fh = logging.FileHandler(filename=log_file, mode='a', encoding='utf-8')
            fh.setLevel(level)
            save_formatter =  logging.Formatter('[%(asctime)s] [%(levelname)s]：%(message)s')
            fh.setFormatter(save_formatter)
            self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger

    def info(self, message: str):
        self.logger.info(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def warnning(self, message: str):
        self.logger.warn(message)

    def error(self, message: str):
        self.logger.error(message)

if __name__ == "__main__":
    log = Logger('test').get_logger()
    log.info('test info')