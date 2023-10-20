from dataclasses import dataclass

@dataclass
class Config():
    host = "0.0.0.0"
    port = 8082

    log = True

    #日志等级：critical|error|warning|info|debug|trace
    log_level = "info"

    #是否启动reload，启用reload多进程无法设置: True | False
    reload = False

    #进程个数，既启动引擎的个数，个数越多消耗内存越大，建议设置为CPU核心个数的1到2倍
    process_worker = 1

    # cuda显卡设置，默认0，其他数值在多显卡情况下生效
    cuda_device_number = 0

    batch_size = 4

    # 特征数量， 5 * 3
    fearure_size = 12

    rnn_hidden_size = 128
    rnn_layers = 2

    secret_key = "dlna6e52cazld5q0z5dqr4dlpf6"

