# 3D_Speaker声纹库

## 1.项目简介

提供了 **声纹注册 / 更新 / 删除 / 匹配** 功能

```
registration/
├── registration/                # 核心代码目录
│   ├── speaker_ops.py           # 声纹操作：注册/更新/删除/匹配
│   ├── faiss_utils.py           # FAISS 索引管理工具
│   ├── audio_utils.py           # 模型加载,音频处理与特征提取
│   └── __init__.py
│
├── embeddings_faiss/            # 已注册用户的声纹向量库
├── embeddings_npy/              # 已注册用户的声纹向量存储目录
├── speakerlab/                  # 3D_speaker相关库
├──registration_api.py           # 对外提供 REST API 服务(未测试)
└── README.md                    # 项目说明文档（本文件）
```

## 2.模块说明

### 2.1 `speaker_ops.py`

- **功能**: 管理声纹向量的注册、更新、删除、匹配。
- **主要函数**:
  - `add_speaker(audio_path,  embedding_model, feature_extractor,spk_name, overwrite=False)`
  - `delete_speaker(spk_name)`
  - `match_speaker(audio_path, embedding_model, feature_extractor, top_k=1)`
  - `speaker_exists(spk_name)`

### 3.2 `faiss_utils.py`

- **功能**: FAISS 索引管理。
- **主要函数**:
  - `init_faiss_index()` → 根据 初始化向量库
  - `rebuild_faiss_index()` → 删除或修改后，重建索引

### 3.3 `audio_utils.py`

- **功能**: 音频预处理与声纹向量提取。

- **主要函数**:

  - `load_audio(path)` → 

    ```
    支持本地文件 & 在线URL
    - 音频: WAV, MP3, FLAC
    - 视频: MP4, AVI, MKV, MOV
    返回: waveform (Tensor, [1, T])
    ```

  - `feature_extractor(waveform)` → 提取声学特征 (fbank/CMVN)

  - `get_embedding(audio_path, embedding_model, feature_extractor)` → 获取归一化 embedding（3D_speaker)

### 3.4 `registration_api.py`

- **功能**: 对外提供 REST API 服务（FastAPI 实现）。
- **API 路径**:
  - `POST /isqa/voiceprint/register` → 注册声纹
  - `PUT /isqa/voiceprint/update` → 更新声纹
  - `DELETE /isqa/voiceprint/delete/{userId}` → 删除声纹
  - `POST /isqa/voiceprint/match` → 匹配声纹

------

## 4. 数据存储

- **声纹向量目录**: `registration/embeddings_npy/`
  - 存储用户声纹向量文件，命名规则 `userId.npy`
- **模型文件目录**: `registration/embeddings_onnx/`
  - 存放声纹识别模型（ONNX 格式）
- **索引文件**: 每次增删改操作后自动重建 FAISS 索引