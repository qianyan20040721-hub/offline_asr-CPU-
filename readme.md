# Funasr+3d_Speaker语音转写与说话人识别系统(CPU单进程版)

##  项目目录

```
offline-asr/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python 依赖列表
├── main.py                   # 程序入口（运行接口）
├── .gitignore                # Git 忽略规则
│
├── speakerlab/               # 3D-Speaker 源码
│
├── embeding_faiss/           # 声纹库目录
│   ├── faiss.index           # 声纹索引文件
│   └── speaker_names.pkl     # 说话人名称映射
│
├── models/                   # 模型文件目录
│   ├── Paramformer/          # ASR 模型配置
│   ├── CAM++/                # 说话人识别模型配置
│   ├── VAD/                  # VAD 模型配置
│   ├── PUNC/                 # 标点模型配置
│   └── macbert4csc/          # 文本纠错模型配置
│
└── ASR/                      # 脚本与工具
    ├── diarization_onnx.py   # 说话人分离/确认脚本
    ├── PROFILE.py            # 日志配置
    └── speaker_ASR_onnx.py   # ASR + 说话人识别 pipeline
```

##  项目简介

本项目实现了一个 **离线语音转写与说话人识别系统**，支持以下功能：

- 🎙️ **ASR 转写**：基于 FunASR ONNX 的 Paraformer模型，实现高精度中文语音识别。

- 🔊 **VAD 语音活动检测**：基于 Fsmn-vad，自动检测有效语音片段，过滤静音与噪声。

- 🗣️ **说话人分离（Diarization）**：利用声纹模型与聚类算法区分不同说话人。

- 👤 **声纹识别（Speaker Verification）**：基于 FAISS 索引库快速比对，识别已注册的说话人。

- ✍️ **文本后处理**：包括数字归一化 (ITN)、标点恢复、语义纠错 (MacBERT)、重复消除。

##  模型方面：

### 🔹FunASR 模型

- **VAD**：`speech_fsmn_vad_zh-cn-16k-common-pytorch` （静态量化 ONNX 版本）
- **标点恢复 (PUNC)**：`punc_ct-transformer_zh-cn-common-vocab272727-pytorch`（静态量化 ONNX 版本）
- **ASR 转写**：`speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`（静态量化 ONNX 版本）

> 模型均可从 [ModelScope](https://github.com/modelscope/FunASR/tree/v0.8.8/funasr/export) 下载或自行导出参考[官方文档](https://github.com/modelscope/FunASR/tree/v0.8.8/funasr/export)

### 🔹 3D-Speaker 模型

- **说话人分离 / 声纹识别**：`speech_campplus_sv_zh_en_16k-common_advanced` （PyTorch 版）

### 🔹 其他组件

- **ITN 数字归一化**：基于 [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)
- **语义纠错**：`macbert4csc-base-chinese` （[Huggingface](https://huggingface.co/shibing624/macbert4csc-base-chinese)）

## 快速开始

### 测试环境


### 依赖

```
pip install -r requirements.txt
```

主要依赖：

- `funasr-onnx` / `onnxruntime` / `pycorrector`
- `torch >= 1.10.1`
- `scikit-learn == 1.0.2`
- `ffmpeg`

## 线程调控

为充分利用多核 CPU，合理设置线程数：

```
NUM_THREADS = 6
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
```

**参数解释：**

- `OMP_NUM_THREADS`：OpenMP 并行线程数（影响 onnxruntime / faiss）
- `OPENBLAS_NUM_THREADS / MKL_NUM_THREADS / VECLIB_MAXIMUM_THREADS`：控制矩阵库并发度
- `NUMEXPR_NUM_THREADS`：`numexpr` 的计算线程
- `OMP_WAIT_POLICY=PASSIVE`：降低空闲线程 CPU 占用

👉 避免 **线程争用 (thread contention)**，提升 CPU-only 环境下的稳定性。

##  核心函数

### 1️⃣ VAD 分段

```
def vad_segment_onnx(wav_path, vad_model):
    wav_list = vad_model.load_data(wav_path)
    wav = wav_list[0]  
    segments = vad_model(wav)  
    return wav, segments
```

- **输入**：音频路径 `wav_path`，VAD 模型 `vad_model`
- **输出**：音频波形 `wav`，分段结果 `segments=[[st,ed],...]`

------

### 2️⃣ 说话人分离 + 声纹识别

```
diar_labels = speaker_diarization_fusion(
    wav, vad_segments, index, names,
    embedding_model, feature_extractor, cluster
)
```

1. **VAD 分段**（输入的 segments）
2. **RMS 动态能量阈值**过滤（去掉太短或太弱的段落）
3. **说话人聚类**（基于 embedding） 
4. **声纹比对**（FAISS 检索已注册说话人库） 

- 最终输出时间戳 + 说话人标签
- 已注册的名字（如 "物业经理"）
- 未注册的聚类标签（如 "speaker[0]"）

------

### 3️⃣ ASR 转写（切片并行）

```
def run_asr(diar_labels, wav_cache):
    # 长段切片 (<=max_chunk_sec)，批量调用 asr_model
    return output_segments, asr_results_batch, chunk_info
```

- **解决长音频问题**：自动切分并保证分片内时长合理。

------

### 4️⃣ 后处理

- **deduplicate_tokens**：去重，解决切片重叠问题
- **merge_output_segments**：合并相邻同说话人短文本
- **itn_process_batch**：数字 / 时间正则化
- **SemanticCorrection (MacBERT)**：语义纠错
- **remove_repetition**：去掉口头语、重复字词

------


## 推理流程

原本说话人分离和声纹识别都需要提取声纹，现使用CAM++_3dspeaker模型将声纹识别流程嵌入说话人分离流程中，声纹提取流程合并仅提取一遍声纹便可以完成说话人分离和声纹识别。

```
 [音频文件] 
     │
     ▼
   VAD 分段
     │
     ▼
说话人分离 / 声纹识别
     │
     ▼ 
  ASR 转写
     │
     ▼
ITN + 标点 + 语义纠错
     │
     ▼
 ✅ 带时间戳的最终文本
```

## 模型速度

**测试环境**

- CPU: Intel Core i7-12700H
- 系统: Windows

| 模型方案              | 音频数量 | 总耗时 (s) | 总时长 (s) | 平均占比 |
| --------------------- | -------: | ---------: | ---------: | -------: |
| 非量化模型 + 语义纠错 |        5 |     472.40 |   11277.95 |    4.19% |
| 量化模型 + 语义纠错   |        5 |     461.88 |   11277.95 |    4.10% |

## 更换模型

| 模型                      | 精度 (WER)      | 特点                                                         | 热词能力  | 长语音/噪声鲁棒性                          | 速度        | 适用场景              |
| ------------------------- | --------------- | ------------------------------------------------------------ | :-------- | ------------------------------------------ | ----------- | --------------------- |
| **Paraformer**            | ⭐⭐⭐⭐ 高         | 基线模型，基于 CIF (Continuous Integrate-and-Fire) 边界预测  | ⭐⭐ 一般   | ⭐⭐ 中等                                    | ⭐⭐⭐⭐ 快     | 短/中长音频           |
| **Contextual Paraformer** | ⭐⭐⭐½ 略低于基线 | 在 Paraformer 基础上增加上下文/热词增强模块                  | ⭐⭐⭐⭐⭐ 强  | ⭐⭐ 中等                                    | ⭐⭐⭐ 中      | 热词/专有名词频繁场景 |
| **SEACO-Paraformer**      | ⭐⭐⭐⭐⭐ 最高      | 引入 **SEACO 模块** (Semantic & Acoustic Offset 校正)，联合声学+语义信息 | ⭐⭐⭐⭐ 较好 | ⭐⭐⭐⭐⭐ 强（长语音、口吃、噪声环境表现最佳） | ⭐⭐ 中等偏慢 | 高精度需求场景        |

