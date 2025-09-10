import os
import numpy as np
from registration.audio_utils import get_embedding
from registration.faiss_utils import (
    embedding_dir,embedding_dir_npy,
    init_faiss_index,
    rebuild_faiss_index
)
from fastapi import HTTPException
def add_speaker(audio_path,embedding_model,feature_extractor, spk_name=None, overwrite=False):
    """添加说话人到 embedding_dir，并重建索引"""
    emb = get_embedding(audio_path,embedding_model,feature_extractor)
    if emb is None:
        return {"success": False, "error": "embedding 提取失败"}

    if spk_name is None:
        spk_name = os.path.splitext(os.path.basename(audio_path))[0]

    npy_path = os.path.join(embedding_dir_npy, f"{spk_name}.npy")

    # ✅ 检查是否已存在
    if os.path.exists(npy_path) and not overwrite:
        print(f"[Add] 添加失败: error: 用户编号已存在")
        return {"success": False, "error": "用户编号已存在"}

    # 保存 embedding
    np.save(npy_path, emb)
    index, names = rebuild_faiss_index()
    print(f"[Add] 添加完成: {spk_name}, 当前库中人数: {len(names)}")
    return {"success": True, "spk_name": spk_name, "count": len(names)}


def speaker_exists(spk_name: str) -> bool:
    """检查某个说话人是否已存在于 embedding_dir"""
    npy_path = os.path.join(embedding_dir_npy, f"{spk_name}.npy")
    return os.path.exists(npy_path)


def delete_speaker(spk_name):
    """删除说话人（删除对应的 .npy 文件），并重建索引"""
    npy_path = os.path.join(embedding_dir_npy, f"{spk_name}.npy")
    if not os.path.exists(npy_path):
        return {"success": False, "error": "用户不存在"}

    os.remove(npy_path)
    print(f"[Delete] 已删除 {spk_name}")
    index, names = rebuild_faiss_index()
    return {"success": True, "spk_name": spk_name, "count": len(names)}


def match_speaker(audio_path, embedding_model, feature_extractor, top_k=1):
    """匹配音频对应的说话人"""
    emb = get_embedding(audio_path, embedding_model, feature_extractor)
    if emb is None:
        return {"success": False, "error": "embedding 提取失败"}

    index, names = rebuild_faiss_index()  # ✅ 保证和 .npy 文件同步
    if len(names) == 0:
        print("[Match] 库中没有任何说话人")
        return {"success": False, "error": "库中没有任何说话人"}

    # FAISS 默认用 L2 距离
    D, I = index.search(np.expand_dims(emb, axis=0), top_k)
    results = []
    for j, i in enumerate(I[0]):
        if i != -1:
            results.append({
                "speaker": names[i],
                "distance": float(D[0][j]),
                "metric": "L2_distance",   # 或者 "cosine_similarity"
                "method": "faiss_index"
            })

    print(f"[Match] 使用方式: faiss_index (L2距离)")
    print(f"[Match] 匹配结果: {results}")

    return {"success": True, "results": results, "method": "faiss_index", "metric": "L2_distance"}

