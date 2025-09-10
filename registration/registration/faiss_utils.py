import os
import pickle
import numpy as np
import faiss
embedding_dir=r"embeddings_faiss/"
embedding_dir_npy = r"embeddings_npy/"
faiss_index_path = os.path.join(embedding_dir, "embeddings_faiss.index")
metadata_path = os.path.join(embedding_dir, "speaker_metadata.pkl")


def init_faiss_index(emb_dim=None,
                     embedding_dir=embedding_dir,
                     npy_dir=embedding_dir_npy):
    """
    初始化 FAISS 索引：
    - 如果已有索引文件，直接加载
    - 否则创建一个空索引（维度由 emb_dim 决定）
    """
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    # ---------- 已有索引 ----------
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "rb") as f:
            names = pickle.load(f)
        print(f"[FAISS] 加载现有索引: {index.ntotal} 个向量，维度 {index.d}")
        return index, names

    # ---------- 没有索引 ----------
    if emb_dim is None:
        print("[FAISS] 没有索引文件，也没有指定 emb_dim，无法初始化，返回 None")
        return None, []

    index = faiss.IndexFlatIP(emb_dim)
    names = []
    print(f"[FAISS] 新建空索引，维度 {emb_dim}")
    return index, names


def rebuild_faiss_index():
    """删除或修改后，重建索引"""
    names = []
    embs = []
    for file in os.listdir(embedding_dir_npy):
        if file.endswith(".npy"):
            spk_name = os.path.splitext(file)[0]
            emb = np.load(os.path.join(embedding_dir_npy, file))
            names.append(spk_name)
            embs.append(emb)

    if len(embs) == 0:
        print("没有可用的 embedding，索引为空")
        return faiss.IndexFlatIP(1), []

    embs = np.vstack(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, faiss_index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(names, f)

    print(f"重建索引完成，共 {len(names)} 个 speaker")
    return index, names
