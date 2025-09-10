import os
import uuid
import json
import requests
import tempfile
import datetime
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from confluent_kafka import Producer

from registration.audio_utils import get_embedding_model
from registration.speaker_ops import add_speaker, delete_speaker, speaker_exists
from registration.faiss_utils import init_faiss_index
import sys
import os

# 将项目根目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 初始化 FAISS 索引
init_faiss_index(emb_dim=512)
embedding_model,feature_extractor= get_embedding_model(device="cpu")
app = FastAPI()

# ========= Kafka Producer =========
producer = Producer({"bootstrap.servers": "localhost:9092"})

async def send_kafka_event_async(event_type: str, data: dict):
    """异步发送 Kafka 消息，不阻塞接口"""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, send_kafka_event, event_type, data)



async def send_kafka_event(event_type: str, data: dict):
    payload = {
        "event": event_type,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "data": data
    }
    try:
        # 仅打印，不发送
        print("[Kafka模拟发送] ", json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        print("[Kafka异常] ", e)

# ========= 请求参数模型 =========
class RegisterRequest(BaseModel):
    userId: str
    audioUrl: str

class UpdateRequest(BaseModel):
    userId: str
    audioUrl: str

@app.post("/isqa/voiceprint/register")
async def register(req: RegisterRequest):
    result = add_speaker(req.audioUrl, embedding_model,feature_extractor,spk_name=req.userId, overwrite=False)

    if not result.get("success", False):
        return {
            "code": 400,
            "message": result.get("error", "声纹注册失败")
        }

    reg_id = str(uuid.uuid4().hex)
    response = {
        "code": 200,
        "message": "声纹注册成功",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "data": {"userId": req.userId, "registrationId": reg_id}
    }

    asyncio.create_task(send_kafka_event_async("voiceprint.register", {**response["data"], "audioUrl": req.audioUrl}))
    return response


@app.put("/isqa/voiceprint/update")
async def update(req: UpdateRequest):
    result = add_speaker(req.audioUrl, embedding_model,feature_extractor, spk_name=req.userId, overwrite=True)

    if not result.get("success", False):
        return {
            "code": 404,
            "message": result.get("error", "声纹更新失败")
        }

    reg_id = str(uuid.uuid4().hex)
    response = {
        "code": 200,
        "message": "声纹更新成功",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "data": {"userId": req.userId, "registrationId": reg_id}
    }

    asyncio.create_task(send_kafka_event_async("voiceprint.update", {**response["data"], "audioUrl": req.audioUrl}))
    return response


@app.delete("/isqa/voiceprint/delete/{userId}")
async def delete(userId: str):
    result = delete_speaker(userId)

    if not result.get("success", False):
        return {
            "code": 404,
            "message": result.get("error", "声纹删除失败")
        }

    response = {
        "code": 200,
        "message": "声纹删除成功",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "data": {"userId": userId}
    }

    asyncio.create_task(send_kafka_event_async("voiceprint.delete", response["data"]))
    return response