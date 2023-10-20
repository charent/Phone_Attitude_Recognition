# auttitude_recognition_server

## 异常姿态识别服务端

## 环境要求：

* python>=3.7，3.7、3.8测试无问题
* cuda>=10.1，建议10.2

# 安装依赖：

```bash
# 安装有网络问题可以替换为阿里源，coda安装依赖自行百度
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host tuna.tsinghua.edu.cn
```


# 运行:

```bash
# 修改config配置
vi config.py

# 运行
python main.py 

# ssl文件生成（非必须）
python ssl/cert.py

# 或者
uvicorn main:app --host 127.0.0.1 --port 8082  --workers 1 --ssl-keyfile ./ssl/key.pem --ssl-certfile ./ssl/cert.pem


```