
import csv
from six import reraise, with_metaclass
import fastapi
import numpy as np
from torch.utils import data

import uvicorn
from fastapi import FastAPI,Depends,status,File,UploadFile,Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.exceptions import HTTPException
from os.path import dirname, abspath
import numpy as np
import torch
import joblib
import os, sys

import psutil

sys.path.append('..')
sys.path.append('.')

from config import Config
from utils.logger import Logger
from model.rnn_model import RNN_Model
from utils.data_process import process_post_file
from utils.function import array_pad, data_sample

# ########################################################3

log = Logger('main', std_out=True).get_logger()
CONFIG  = Config()
app = FastAPI(docs_url=None, redoc_url=None, version="1.1.0")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

# 设置跨域
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_middleware(HTTPSRedirectMiddleware)

base_path = abspath(dirname(__file__))

device = torch.device("cpu")
# if torch.cuda.is_available():
    # device = torch.device("cuda:{}".format(CONFIG.cuda_device_number))
    # torch.backends.cudnn.benchmark = True
# log.info('device: {}'.format(device))


# 深度学习模型
gru_model = RNN_Model(
    feature_size=CONFIG.fearure_size,
    rnn_layers=CONFIG.rnn_layers,
    hidden_size=CONFIG.rnn_hidden_size,
    output_class=3,
).to(device)
gru_model.eval()

# 机器学习模型
liner_svc = joblib.load('{}/model_file/svm.pkl'.format(base_path))

gru_model.load_state_dict(torch.load("{}/model_file/gru.pkl".format(base_path), map_location=device))
gru_model.eval()

CLASS_INFO = {
    0: "正常操作",
    1: "输入中换人",
    2: "行走输入"
}

INFO = {
    # xxx场景
    0: {
        0:{"text": "正常操作", "risk": 0},
        1: {"text": "输入中换人，存在xxxx风险", "risk": 1},
        2: {"text": "行走输入，疑似xxxx", "risk": 1},
    },

    # xxxx场景
    1:{
        0: {"text": "正常操作", "risk": 0},
        1: {"text": "输入中换人，疑似风险xxxx", "risk": 1},
        2: {"text": "xxxx时行走输入，疑似风险xxxxx", "risk": 1},
    }
}


async def api_key_auth(token: str = Depends(oauth2_scheme)):
    """
    验证post请求的key是否和服务器的key一致
    需要在请求头加上 Authorization: Bearer SECRET_KEY
    """
    if token == CONFIG.secret_key:
        return None

    # 验证出错
    raise HTTPException(
        status_code= status.HTTP_401_UNAUTHORIZED,
        detail="api认证未通过，请检查认证方式和token！",
        headers={"WWW-Authenticate": "Bearer"},
    )



class RCjson(BaseModel):
    scene_id: int

@app.post('/recognition')
async def recognition(scene_id: int=Form(...), file: UploadFile = File(...), authority: str = Depends(api_key_auth)):
    
    # content = await file.file.
    lines = [str(line, encoding='utf-8') for line in file.file.readlines() ]

    # with open("./123.csv", 'w', encoding='utf-8') as f:
    #     f.writelines(lines)

    cpu_use =  float(psutil.cpu_percent(interval=0))

    if cpu_use > 85.0:
        # CPU使用率小于85的时候使用机器学习模型，数据不需要正则化
        try:
            data = process_post_file(lines, normalization=False)
        except Exception as e:
            return {'status': 'fail', 'message': str(e)}

        data = data_sample(data, sample_interval=15, max_column=768)

        pred_y =  liner_svc.predict([data])[0]

        attitude = INFO[scene_id][pred_y]['text']
        risk = INFO[scene_id][pred_y]['risk']
        print("svm:", attitude, end=' ')

        return {"status": "success", "risk": risk, "attitude": attitude, "file_name": file.filename}

    # 深度学习模型需要正则化
    try:
        data = process_post_file(lines, normalization=True)
    except Exception as e:
        return {'status': 'fail', 'message': str(e)}

    data_x, lengths = array_pad([data], return_length=True)

    as_tensor = torch.as_tensor

    data_x = as_tensor(data_x, dtype=torch.float32).to(device)
    lengths = as_tensor(lengths, dtype=torch.long)
    
    with torch.no_grad():
        pred_y = gru_model(data_x, lengths)
        pred_y = torch.argmax(pred_y, dim=-1).detach().cpu().numpy()[0]

    attitude = INFO[scene_id][pred_y]['text']
    risk = INFO[scene_id][pred_y]['risk']
    print("gru:", attitude, end=' ')

    return {"status": "success", "risk": risk, "attitude": attitude, "file_name": file.filename}


@app.get("/download_apk")
def download_apk():
    return FileResponse('./apk/app-release.apk', filename='auttotude_recognition.apk')

@app.post("/check_update")
def check_update():
    with open('./apk/version', 'r', encoding='utf-8') as f:
        version = f.readline().strip()

    return {"version": version}

if __name__ == '__main__':

    # 加上reload参数（reload=True）时，多进程设置无效
    workers = max(CONFIG.process_worker, 1)
    log.info('启动的进程个数:{}'.format(workers))

    # uvicorn main:app --host 127.0.0.1 --port 8082  --workers 1 --ssl-keyfile ./ssl/key.pem --ssl-certfile ./ssl/cert.pem

    uvicorn.run(
        'main:app',
        host=CONFIG.host,
        port=CONFIG.port,
        reload=CONFIG.reload,
        workers=workers,
        log_level=CONFIG.log_level,
        # ssl_keyfile="./ssl/key.pem",
        # ssl_certfile="./ssl/cert.pem",
    )
