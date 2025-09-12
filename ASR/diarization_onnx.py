import torch
import numpy as np
from pyannote.core import SlidingWindowFeature

from speakerlab.bin.infer_diarization import compressed_seg
from speakerlab.utils.builder import build
from speakerlab.utils.config import Config
from speakerlab.utils.utils import circle_pad, merge_vad
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pyannote.audio import Model, Inference
from scipy import optimize

def get_embedding_model(device=None):
    conf = {
        'model_id': r"C:\Users\ROG\.cache\modelscope\hub\models\iic\speech_campplus_sv_zh-cn_3dspeaker_16k",
        'revision': 'v1.0.0',
        'model_ckpt': r'C:\Users\ROG\.cache\modelscope\hub\models\iic\speech_campplus_sv_zh-cn_3dspeaker_16k\campplus_cn_3dspeaker.bin',
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


def get_cluster_backend():
    conf = {
        'cluster': {
            'obj': 'speakerlab.process.cluster.CommonClustering',
            'args': {'cluster_type': 'spectral', 'mer_cos': 0.8, 'min_num_spks': 1,
                     'max_num_spks': 15, 'min_cluster_size': 4, 'oracle_num': None, 'pval': 0.012}
        }
    }
    config = Config(conf)
    return build('cluster', config)


def get_segmentation_model(hf_token, device=None):
    model = Model.from_pretrained('pyannote/segmentation-3.0', use_auth_token=hf_token, strict=False)
    segmentation = Inference(model, duration=model.specifications.duration,
                             step=0.1 * model.specifications.duration, skip_aggregation=True,
                             batch_size=32, device=device)
    return segmentation



def _build_chunks_from_segments(segments, win=1.5, step=0.75):
    chunks = []
    for st, ed in segments:
        sub_st = st
        while sub_st < ed:
            sub_ed = min(sub_st + win, ed)
            if sub_ed - sub_st > 1e-3:
                chunks.append([sub_st, sub_ed])
            sub_st += step
    return chunks

def _accumulate_activity_on_grid(windows, labels, grid_starts, grid_dur):
    """
    根据 (window, cluster_label) 在帧网格上累计每个 cluster 的占用强度
    windows: [[st, ed], ...]
    labels:  [k0, k1, ...]  (0..K-1)
    grid_starts: shape [T], 每一帧窗口起点（秒）
    grid_dur: float, 帧窗口时长（秒）
    return: activity [K, T]，强度∈[0,1]
    """
    K = int(max(labels)) + 1 if len(labels) else 0
    T = len(grid_starts)
    if K == 0 or T == 0:
        return np.zeros((0, T), dtype=np.float32)

    act = np.zeros((K, T), dtype=np.float32)
    frame_ends = grid_starts + grid_dur

    # 对每个窗口把覆盖到的帧加分（按窗口与帧的重叠占比）
    for (wst, wed), k in zip(windows, labels):
        # 只处理有效窗口
        if wed <= wst:
            continue
        # 粗筛：二分/索引以减少遍历
        # 找到可能相交的帧索引范围
        i0 = np.searchsorted(frame_ends, wst, side="right") - 1
        i0 = max(i0, 0)
        i1 = np.searchsorted(grid_starts, wed, side="left")
        i1 = min(i1, T)

        for i in range(i0, i1):
            fst, fed = grid_starts[i], frame_ends[i]
            ov = max(0.0, min(wed, fed) - max(wst, fst))
            if ov > 0:
                act[k, i] += ov / grid_dur  # 归一化到[0, 1]强度

    # 轻微平滑（3 帧均值滤波）
    if T >= 3:
        kernel = np.array([1, 1, 1], dtype=np.float32) / 3.0
        for k in range(K):
            act[k] = np.convolve(act[k], kernel, mode="same")

    # 裁剪到 [0,1]
    act = np.clip(act, 0.0, 1.0)
    return act

def post_process(binary, count=None):
    """
    根据 count 约束每帧活跃说话人数
    binary: [num_frames, num_speakers]
    count:  [num_frames] 每帧允许的活跃人数
    """
    num_frames, num_spks = binary.shape
    sorted_spks = np.argsort(-binary, axis=-1)
    final = np.zeros_like(binary)
    for t in range(num_frames):
        k = int(count[t]) if count is not None else 1
        k = min(k, num_spks)
        for s in sorted_spks[t, :k]:
            if binary[t, s] > 0:
                final[t, s] = 1.0
    return final


def compute_rms(x: np.ndarray):
    """计算归一化音频的 RMS"""
    return np.sqrt(np.mean(np.square(x)))

def get_dynamic_rms_threshold(wav: np.ndarray, base_ratio: float = 0.3, min_thresh: float = 0.000005):
    """
    根据整段音频的 RMS 动态计算阈值
    :param wav: 归一化后的音频数据 (float32, -1~1)
    :param base_ratio: 相对比率, 例如 0.3 表示阈值取全局 RMS 的 30%
    :param min_thresh: 最低阈值, 防止阈值过低
    """
    global_rms = compute_rms(wav)
    dyn_thresh = max(global_rms * base_ratio, min_thresh)
    return dyn_thresh, global_rms


from pyannote.core import SlidingWindow
def speaker_diarization_fusion(
        wav, segments, index, names, device='cpu',
        embedding_model=None, feature_extractor=None,
        cluster=None,
        speaker_num=None,
        faiss_threshold=0.55,  # FAISS 相似度阈值
):
    """
    融合版说话人 diarization:
    - 先聚类 -> 得到 speaker[cluster_id]
    - 合并相邻相同 cluster_id 的片段
    - 对合并后的 embedding 做 FAISS 查询
    - 已知人替换名字，未知人保留 speaker[id]
    """
    fs = feature_extractor.sample_rate
    batchsize = 64
    min_dur = 0.7
    chunks, wavs = [], []
    dur, step = 1.5, 0.75
    dyn_thresh, global_rms = get_dynamic_rms_threshold(wav, base_ratio=0.3)

    # print(f"全局 RMS = {global_rms:.4f}, 动态阈值 = {dyn_thresh:.4f}")

    # --- 分块 ---
    for st, ed in segments:
        seg_wav = wav[int(st * fs):int(ed * fs)]
        if len(seg_wav) == 0:
            continue
        seg_rms = compute_rms(seg_wav)

        if seg_rms >= dyn_thresh and (ed - st) >= min_dur:
            sub_st = st
            while sub_st + dur < ed + step:
                sub_ed = min(sub_st + dur, ed)
                chunks.append([sub_st, sub_ed])
                wavs.append(wav[int(sub_st * fs):int(sub_ed * fs)])
                sub_st += step
        else:
            # print(f"丢弃 segment [{st:.2f}, {ed:.2f}] RMS={seg_rms:.4f} < 阈值={dyn_thresh:.4f}")
            continue

    # --- 统一 padding ---
    if len(wavs) > 0:
        max_len = max([x.shape[0] for x in wavs])
        wavs = [circle_pad(torch.from_numpy(x), max_len) for x in wavs]
        wavs = torch.stack(wavs).unsqueeze(1)
    else:
        return []

    # --- 提取 embedding ---
    embeddings = []
    batch_st = 0
    device = torch.device(device)
    with torch.no_grad():
        while batch_st < len(chunks):
            wav_batch = wavs[batch_st: batch_st + batchsize].to(device)
            feats_batch = torch.vmap(feature_extractor)(wav_batch)
            model_dtype = next(embedding_model.parameters()).dtype
            feats_batch = feats_batch.to(model_dtype)
            emb_batch = embedding_model(feats_batch).cpu()
            embeddings.append(emb_batch)
            batch_st += batchsize
    embeddings = torch.cat(embeddings, dim=0).numpy()

    # --- 聚类 ---
    cluster_labels = cluster(embeddings, speaker_num=speaker_num if speaker_num else None)
    diar_labels = [[st, ed, int(spk)] for (st, ed), spk in zip(chunks, cluster_labels)]

    # --- 合并相邻同 cluster 的片段 ---
    diar_labels = compressed_seg(diar_labels)  # e.g. [[st, ed, cluster_id], ...]

    # --- 直接对合并片段做 embedding 平均 + FAISS 搜索 ---
    diar_labels_with_name = []
    for st, ed, cl in diar_labels:
        # 取属于该 cluster 的所有 chunk embedding
        emb_idxs = [i for i, cid in enumerate(cluster_labels) if cid == cl]
        if not emb_idxs:
            print(f"[WARN] cluster {cl} 在区间 ({st:.2f}, {ed:.2f}) 内没有找到 chunk embedding")
            continue

        seg_emb = embeddings[emb_idxs].mean(axis=0)
        seg_emb = seg_emb / np.linalg.norm(seg_emb)

        scores, ids = index.search(np.expand_dims(seg_emb, 0), 1)
        score = scores[0][0]
        match_id = ids[0][0]

        if score > faiss_threshold:
            speaker = names[match_id]
        else:
            speaker = f"speaker[{cl}]"

        diar_labels_with_name.append([st, ed, speaker])

    return diar_labels_with_name
