# auttitude_recognition
## 基于pytorch的时间序列分类模型

## 环境要求：
* python>=3.7，3.7、3.8测试无问题
* cuda>=10.1，建议10.2（非必须）

## 安装依赖：

```bash
# 安装有网络问题可以替换为阿里源，coda安装依赖自行百度
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host tuna.tsinghua.edu.cn
```

## 替换模型

- pytorch训练模型，替换掉main.py定义模型和加载模型部分
- 其他模型，在main.py引入模型代码，如果提示缺少xx库（包），按照提示按照即可，最后重写定义模型和加载模型部分
- process_post_file函数返回数组的形状：（time_seqence,15）,time_seqence为时间序列的长度

## 运行:

```bash

# 修改config配置（非必须）
vi config.py

# 运行
python main.py
    
```

## APP测试模型效果

- 点击app界面的右上角的三个点，修改服务器为自定义服务器，IP地址输入本机的IP地址，连接了WiFi则为WiFi的地址，一般为192.168.x.x的格式，如果在配置文件config.py中修改了端口号，设置里的端口号也得改 ，保存设置。

- 在app识别界面测试模型的泛化性能即可