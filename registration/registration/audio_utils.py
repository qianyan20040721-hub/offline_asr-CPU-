import io
import os

import requests
import torch
import torchaudio
import numpy as np
import onnxruntime as ort
from pydub import AudioSegment

from speakerlab.utils.builder import build
from speakerlab.utils.config import Config


def get_embedding_model(device=None):
    conf = {
        'model_id': r"speech_campplus_sv_zh-cn_3dspeaker_16k",
        'revision': 'v1.0.0',
        'model_ckpt': r'speech_campplus_sv_zh-cn_3dspeaker_16k\campplus_cn_3dspeaker.bin',
        'embedding_model': {
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {'feat_dim': 80, 'embedding_size': 512}
        },
        'feature_extractor': {
            'obj': 'speakerlab.process.processor.FBank',
            'args': {'n_mels': 80, 'sample_rate': 16000, 'mean_nor': True}
        }
    }
    config = Config(conf)
    feature_extractor = build('feature_extractor', config)
    embedding_model = build('embedding_model', config)
    pretrained_state = torch.load(conf['model_ckpt'], map_location='cpu')
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    if device:
        embedding_model.to(device)
    return embedding_model, feature_extractor

device= "cpu"




def load_audio(audio_path, target_sr=16000):
    """
    支持本地文件 & 在线URL
    - 音频: WAV, MP3, FLAC
    - 视频: MP4, AVI, MKV, MOV
    返回: waveform (Tensor, [1, T])
    """
    # ---------- 在线URL ----------
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        response = requests.get(audio_path, stream=True)
        response.raise_for_status()
        data = io.BytesIO(response.content)  # 不落地，直接放内存

        # pydub自动识别格式（视频/音频）
        audio = AudioSegment.from_file(data)
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (1 << 15)
        waveform = torch.from_numpy(samples).unsqueeze(0)
        return waveform

    # ---------- 本地文件 ----------
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in [".wav", ".mp3", ".flac"]:
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        return waveform

    elif ext in [".mp4", ".avi", ".mkv", ".mov"]:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (1 << 15)
        waveform = torch.from_numpy(samples).unsqueeze(0)
        return waveform

    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def get_embedding(audio_path,embedding_model,feature_extractor):
    waveform = load_audio(audio_path)
    feat = feature_extractor(waveform)   # 和匹配保持一致
    feat = feat.to(next(embedding_model.parameters()).dtype)
    with torch.no_grad():
        emb = embedding_model(feat.unsqueeze(0))  # [1, emb_dim]
    emb = emb.squeeze(0).cpu().numpy()
    emb = emb / np.linalg.norm(emb)
    return emb.astype(np.float32)