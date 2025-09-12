import os
import pickle

import faiss
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional

from funasr_onnx import Paraformer, CT_Transformer, Fsmn_vad
from itn.chinese.inverse_normalizer import InverseNormalizer

from ASR.PROFILE import logger
from ASR.diarization_onnx import get_embedding_model, get_cluster_backend
from ASR.speaker_ASR_onnx import SemanticCorrection,  transcribe_with_speaker_parallel

# 假设这些是必要的导入
# 实际使用时请根据你的库进行调整

FAISS_INDEX_PATH = "embeding_faiss/faiss.index"
METADATA_PATH = "embeding_faiss/speaker_names.pkl"

# 配置常量
NUM_THREADS = 2  # 根据实际情况调整


def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise ValueError("FAISS 索引或元数据不存在，请先用 init_faiss_index 创建")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        names = pickle.load(f)
    logger.info("加载 FAISS 索引: %d 个向量", index.ntotal)
    return index, names

class SpeechProcessor:
    """语音处理主类，封装了所有语音处理相关的功能"""

    def __init__(
            self,
            model_dir_asr: str,
            model_dir_punc: str,
            model_dir_vad: str,
            model_dir_cor: str,
            device: str = "cpu",
            num_threads: int = NUM_THREADS
    ):
        """
        初始化语音处理器

        Args:
            model_dir_asr: ASR模型目录
            model_dir_punc: 标点模型目录
            model_dir_vad: VAD模型目录
            model_dir_cor: 语义纠错模型目录
            device: 运行设备，默认为"cpu"
            num_threads: 线程数量，默认为4
        """
        self.num_threads = num_threads
        self.device = device

        # 初始化ONNX会话选项
        self.session_options = self._init_session_options()

        # 初始化各种模型
        self.invnormalizer = InverseNormalizer(overwrite_cache=True)
        self.asr_model = self._init_asr_model(model_dir_asr)
        self.punc_model = self._init_punc_model(model_dir_punc)
        self.vad_model = self._init_vad_model(model_dir_vad)
        self.sem_correction_model = self._init_correction_model(model_dir_cor)

        # 初始化嵌入模型和聚类相关组件
        self.embedding_model, self.feature_extractor = self._init_embedding_model()
        self.cluster = self._init_cluster_backend()
        self.index, self.names = self._load_faiss_index()

    def _init_session_options(self) -> ort.SessionOptions:
        """初始化ONNX会话选项"""
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.num_threads
        so.inter_op_num_threads = 1
        return so

    def _init_asr_model(self, model_dir: str) -> Paraformer:
        """初始化ASR模型"""
        return Paraformer(
            model_dir,
            batch_size=4,
            quantize=True,
            intra_op_num_threads=self.num_threads
        )

    def _init_punc_model(self, model_dir: str) -> CT_Transformer:
        """初始化标点模型"""
        return CT_Transformer(
            model_dir,
            batch_size=4,
            quantize=True,
            intra_op_num_threads=self.num_threads
        )

    def _init_vad_model(self, model_dir: str) -> Fsmn_vad:
        """初始化VAD模型"""
        return Fsmn_vad(
            model_dir,
            batch_size=1,
            quantize=True,
            intra_op_num_threads=self.num_threads,
            max_end_sil=1600
        )

    def _init_correction_model(self, model_dir: str) -> SemanticCorrection:
        """初始化语义纠错模型"""
        return SemanticCorrection(model_dir)

    def _init_embedding_model(self) -> Tuple[object, object]:
        """初始化嵌入模型"""
        return get_embedding_model(self.device)

    def _init_cluster_backend(self) -> object:
        """初始化聚类后端"""
        return get_cluster_backend()

    def _load_faiss_index(self) -> Tuple[object, List[str]]:
        """加载FAISS索引"""
        return load_faiss_index()

    def process_audio_file(
            self,
            audio_path: str
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        处理单个音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            包含语音片段信息的列表和时间统计字典
        """
        return transcribe_with_speaker_parallel(
            audio_path,
            self.vad_model,
            self.asr_model,
            self.punc_model,
            self.embedding_model,
            self.feature_extractor,
            self.cluster,self.index,self.names,
            sem_correction_model=self.sem_correction_model,
        )

    def process_audio_directory(
            self,
            audio_dir: str,
            max_files: Optional[int] = None
    ) -> Tuple[List[Dict], float]:
        """
        处理音频目录中的所有文件

        Args:
            audio_dir: 音频目录路径
            max_files: 最大处理文件数，None表示处理所有文件

        Returns:
            所有音频的处理结果和总耗时
        """
        # 获取所有音频文件
        audio_files = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith((".flac", ".wav"))
        ]

        # 限制最大处理文件数
        if max_files is not None and max_files > 0:
            audio_files = audio_files[:max_files]

        total_timings = 0.0
        all_results = []

        # 处理每个音频文件
        for idx, audio_path in enumerate(audio_files, 1):
            print(f"===== 第 {idx} 个文件: {audio_path} =====")
            output_segments, timings = self.process_audio_file(audio_path)

            # 保存结果
            all_results.extend(output_segments)
            total_timings += timings.get("Total", 0.0)

            # 打印当前文件的结果
            self._print_processing_results(output_segments, timings)

        return all_results, total_timings

    def _print_processing_results(
            self,
            segments: List[Dict],
            timings: Dict[str, float]
    ) -> None:
        """打印处理结果"""
        for seg in segments:
            token_strs = []
            for t in seg.get("token", []):
                char = t[0]
                start, end = t[1]
                token_strs.append(f"{char}({start:.2f}-{end:.2f})")
            tokens_fmt = " ".join(token_strs)
            print(f"[Speaker {seg['speaker']}] {seg['text']}")
            print(f"    Tokens: {tokens_fmt}")

        print(f"处理时间: {timings.get('Total', 0.0):.2f}秒")


def main():
    """主函数，演示如何使用SpeechProcessor类"""
    # 模型目录配置
    model_dirs = {
        "asr": r"models\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "punc": r"models\punc_ct-transformer_zh-cn-common-vocab272727-onnx",
        "vad": r"models\speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "cor": r"models\macbert4csc-base-chinese\snapshots\e45934f0e6d7459aa001ff0fe59a23e9b6b786d5"
    }

    # 初始化语音处理器
    processor = SpeechProcessor(
        model_dir_asr=model_dirs["asr"],
        model_dir_punc=model_dirs["punc"],
        model_dir_vad=model_dirs["vad"],
        model_dir_cor=model_dirs["cor"]
    )

    # 音频目录
    audio_dir = r"F:\path\to\aishell4\test\wav"

    # 处理音频文件，最多处理10个
    _, total_timings = processor.process_audio_directory(audio_dir, max_files=10)

    print(f"\n所有文件总处理时间: {total_timings:.2f}秒")


if __name__ == "__main__":
    main()
