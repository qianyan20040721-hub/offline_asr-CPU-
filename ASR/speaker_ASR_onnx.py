#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time

import difflib
from typing import Any, Dict, List, Sequence
import numpy as np
from ASR.PROFILE import logger, PROFILE_STATS
from ASR.diarization_onnx import  speaker_diarization_fusion
from pycorrector.macbert.macbert_corrector import MacBertCorrector

# =========================
# 基本配置
# =========================
NUM_THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"


SAMPLE_RATE = 16000




# =========================
# 语义纠错（MacBert）类（保持你原有实现）
# =========================
class SemanticCorrection:
    def __init__(self, model_name_or_path: str = "shibing624/macbert4csc-base-chinese"):
        self.corrector = MacBertCorrector(model_name_or_path)
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        self.non_chinese_pattern = re.compile(r'[^\u4e00-\u9fff]+')

    def batch_correct(self, texts: Sequence[str], debug: bool = True) -> List[str]:
        corrected_texts: List[str] = []
        for t in texts:
            try:
                res = self.corrector.correct(t)
            except Exception as e:
                if debug:
                    logger.warning("MacBert correct error for text=%s exc=%s", t, e)
                corrected_texts.append(t)
                continue

            if isinstance(res, dict) and "target" in res:
                corrected_t = res["target"]

                non_chinese_orig = ''.join(self.non_chinese_pattern.findall(t))
                non_chinese_corr = ''.join(self.non_chinese_pattern.findall(corrected_t))
                if non_chinese_orig != non_chinese_corr:
                    if debug:
                        logger.warning("非中文部分被修改，回退原文: %s", t)
                    corrected_texts.append(t)
                    continue

                corrected_texts.append(corrected_t)
            else:
                if debug:
                    logger.warning("MacBert returned unexpected result, keep original: %s", t)
                corrected_texts.append(t)

        return corrected_texts


# =========================
# 分词 / Token 更新 / ITN / 重复 / 去重 等函数
# 保持原逻辑，仅做小幅规范化
# =========================
def tokenize_mixed(text: str) -> List[str]:
    tokens, buf = [], []
    for ch in text:
        if 'a' <= ch.lower() <= 'z':
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def update_tokens_with_itn(segments: List[Dict[str, Any]], texts_itn: Sequence[str]) -> List[Dict[str, Any]]:
    for idx, (seg, itn_text) in enumerate(zip(segments, texts_itn)):

        if not itn_text.strip():
            continue
        old_tokens = [tok for tok, _ in seg.get("token", [])]
        new_tokens = tokenize_mixed(itn_text)
        matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens)
        new_token_map = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            old_chunk = old_tokens[i1:i2]
            new_chunk = new_tokens[j1:j2]

            if tag == "equal":
                for tok in old_chunk:
                    ts = next((ts for t, ts in seg["token"] if t == tok), [-1, -1])
                    new_token_map.append([tok, ts])
            elif tag == "replace":
                ref_ts = next((ts for t, ts in seg["token"] if t in old_chunk), [-1, -1])
                for tok in new_chunk:
                    new_token_map.append([tok, ref_ts])
            elif tag == "insert":
                for tok in new_chunk:
                    new_token_map.append([tok, [-1, -1]])
            elif tag == "delete":
                continue

        seg["text"] = itn_text
        seg["token"] = new_token_map

    return segments


def update_tokens_with_corr_and_punc(output_segments: List[Dict[str, Any]], corrected_texts: Sequence[str], punc_texts: Sequence[str], debug: bool = True) -> List[Dict[str, Any]]:
    def tokenize_mixed_inner(text: str) -> List[str]:
        tokens, buf = [], []
        for ch in text:
            if 'a' <= ch.lower() <= 'z' or '0' <= ch <= '9':
                buf.append(ch)
            else:
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                tokens.append(ch)
        if buf:
            tokens.append("".join(buf))
        return tokens

    for seg, corr_text, punc_text in zip(
        [seg for seg in output_segments if seg.get("text")],
        corrected_texts,
        punc_texts
    ):
        final_text = punc_text if punc_text else corr_text
        final_text = remove_repetition(final_text)

        old_tokens = [tok for tok, _ in seg.get("token", [])]
        final_tokens = tokenize_mixed_inner(final_text)
        matcher = difflib.SequenceMatcher(None, old_tokens, final_tokens)

        if debug:
            logger.debug("[Token-Update] 原始: %s → 最终: %s", old_tokens, final_tokens)

        new_token_map = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            old_chunk = old_tokens[i1:i2]
            new_chunk = final_tokens[j1:j2]

            if tag == "equal":
                for tok in old_chunk:
                    ts = next((ts for t, ts in seg["token"] if t == tok), [-1, -1])
                    new_token_map.append([tok, ts])
            elif tag == "replace":
                is_numeric_replace = any(ch.isdigit() for tok in new_chunk for ch in tok) and any('\u4e00' <= ch <= '\u9fff' for tok in old_chunk for ch in tok)
                if is_numeric_replace:
                    ts_list = [tok_ts for tok, tok_ts in seg.get("token", []) if tok in old_chunk]
                    if ts_list:
                        ts = [min(t[0] for t in ts_list), max(t[1] for t in ts_list)]
                    else:
                        ts = [-1, -1]
                        if debug:
                            logger.debug("数字替换无匹配 token, 使用 ts=%s 对 %s", ts, new_chunk)
                    for tok in new_chunk:
                        new_token_map.append([tok, ts])
                else:
                    ref_ts = next((ts for t, ts in seg.get("token") if t in old_chunk), [-1, -1])
                    for tok in new_chunk:
                        new_token_map.append([tok, ref_ts])
            elif tag == "insert":
                if debug:
                    logger.debug("  [新增Token] %s", [(tok, [-1, -1]) for tok in new_chunk])
                for tok in new_chunk:
                    new_token_map.append([tok, [-1, -1]])
            elif tag == "delete":
                if debug:
                    for tok in old_chunk:
                        ts = next((ts for t, ts in seg.get("token", []) if t == tok), [-1, -1])
                        logger.debug("  [删除Token] (%s, %s)", tok, ts)
                continue

        seg["text"] = final_text
        seg["token"] = new_token_map

    return output_segments


def itn_process_batch(texts: Sequence[str], lang: str = "zh") -> List[str]:
    if lang == "zh":
        # 使用全局 invnormalizer（与原来行为保持一致）
        return [invnormalizer.normalize(t.replace(" ", "")) if t else "" for t in texts]
    else:
        raise NotImplementedError(f"ITN not implemented for {lang}")


def deduplicate_tokens(output_segments: List[Dict[str, Any]], overlap_threshold: float = 0.2) -> List[Dict[str, Any]]:
    cleaned_segments: List[Dict[str, Any]] = []
    last_tokens: List[List[float]] = []

    for seg in output_segments:
        timestamps = seg.get("timestamp", [])
        text = seg.get("text", "")

        # 如果没有时间戳，直接保留
        if not timestamps or len(timestamps) != len(text):
            cleaned_segments.append(seg)
            continue

        new_text = []
        new_timestamps = []
        for ch, ts in zip(text, timestamps):
            is_dup = False
            for prev_ts in last_tokens[-5:]:
                overlap = max(0, min(prev_ts[1], ts[1]) - max(prev_ts[0], ts[0]))
                duration = ts[1] - ts[0]
                if duration > 0 and overlap / duration > overlap_threshold:
                    is_dup = True
                    break
            if not is_dup:
                new_text.append(ch)
                new_timestamps.append(ts)

        if new_text:
            seg["text"] = "".join(new_text)
            seg["timestamp"] = new_timestamps
            cleaned_segments.append(seg)
            last_tokens.extend(new_timestamps)
    return cleaned_segments


def filter_short_segments(segments: Sequence[Sequence[float]], min_dur: float = 0.7) -> List[List[float]]:
    filtered = []
    for seg in segments:
        st, ed = seg[:2]
        if ed - st < min_dur:
            continue
        filtered.append([st, ed])
    return filtered


def vad_segment_onnx(wav_path: str, vad_model: Any):
    wav_list = vad_model.load_data(wav_path)
    wav = wav_list[0]
    segments = vad_model(wav)
    return wav, segments


def merge_short_segments(diar_labels: List[List[Any]], max_gap: float = 3.0) -> List[List[Any]]:
    if not diar_labels:
        return []
    merged = [list(diar_labels[0])]
    for st, ed, spk in diar_labels[1:]:
        prev_st, prev_ed, prev_spk = merged[-1]
        if spk == prev_spk and st - prev_ed <= max_gap:
            merged[-1][1] = ed
        else:
            merged.append([st, ed, spk])
    return merged


def add_tokens(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_segments = []
    for seg in segments:
        text = seg.get("text", "")
        timestamps = seg.get("timestamp", [])
        tokens = text.split()

        token_map = []
        i = 0
        while i < len(tokens) and i < len(timestamps):
            tok, ts = tokens[i], timestamps[i]
            if re.fullmatch(r"[A-Z]", tok):
                buffer = [tok]
                j = i + 1
                while j < len(tokens) and j < len(timestamps) and re.fullmatch(r"[A-Z]", tokens[j]):
                    buffer.append(tokens[j])
                    j += 1
                merged_word = "".join(buffer)
                token_map.append([merged_word, timestamps[j-1]])
                i = j
            else:
                token_map.append([tok, ts])
                i += 1

        seg["token"] = token_map
        new_segments.append(seg)

    return new_segments


def merge_output_segments(output_segments: List[Dict[str, Any]], max_gap: float = 1.5) -> List[Dict[str, Any]]:
    if not output_segments:
        return []

    merged = [output_segments[0]]
    for seg in output_segments[1:]:
        prev = merged[-1]
        prev_end = prev["timestamp"][-1][1] if prev.get("timestamp") else None
        seg_start = seg["timestamp"][0][0] if seg.get("timestamp") else None
        gap = (seg_start - prev_end) if (prev_end is not None and seg_start is not None) else None

        if seg["speaker"] == prev["speaker"] and (gap is None or gap <= max_gap):
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
            prev["timestamp"].extend(seg["timestamp"])
        else:
            merged.append(seg)

    return merged


def remove_repetition(text: str) -> str:
    text = re.sub(r'([\u4e00-\u9fa5])\1{2,}', r'\1', text)
    text = re.sub(r'(([\u4e00-\u9fa5]{2,5}))(\1)+', r'\1', text)
    common_repeats = ["但是", "就是说", "比如说", "然后", "就是"]
    for word in common_repeats:
        text = re.sub(f'({word})+', word, text)
    text = re.sub(r'([。！？…])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =========================
# 主流程（保持你原来签名和调用方式，逻辑不变）
# =========================
def transcribe_with_speaker_parallel(
    audio_path,
    vad_model,
    asr_model,
    punc_model,
    embedding_model,
    feature_extractor,
    cluster,index, names,
    sem_correction_model=None,
):
    total_start = time.time()
    sr = SAMPLE_RATE

    # 1. VAD
    t0 = time.time()
    wav, segments_list = vad_segment_onnx(audio_path, vad_model)

    # 3. VAD 段落转换
    vad_segments = [[s / 1000, e / 1000] for s, e in segments_list[0]]
    vad_segments = filter_short_segments(vad_segments, min_dur=0.7)

    # 记录 profile（安全写入）
    PROFILE_STATS.setdefault("VAD", []).append(time.time() - t0)
    logger.info("VAD : %.3fs, vad_segments=%d", PROFILE_STATS.get("VAD", [0.0])[-1], len(vad_segments))

    # 2. 说话人分离
    t1 = time.time()
    diar_labels = speaker_diarization_fusion(
        wav,
        vad_segments,
        index, names,
        embedding_model=embedding_model,
        feature_extractor=feature_extractor,
        cluster=cluster
    )
    diar_labels = merge_short_segments(diar_labels, max_gap=15)
    PROFILE_STATS.setdefault("Diarization", []).append(time.time() - t1)
    logger.info("Diarization : %.3fs, diar_segments=%d", PROFILE_STATS.get("Diarization", [0.0])[-1], len(diar_labels))

    # 预切片缓存
    wav_cache: Dict[int, np.ndarray] = {}
    for idx, (st, ed, spk) in enumerate(diar_labels):
        wav_cache[idx] = wav[int(st * sr): int(ed * sr)]

    # ASR 处理（分批）
    def run_asr(diar_labels_local, wav_cache_local, sr_local=16000, max_chunk_sec_local=25,
                overlap_sec=0.5, batch_size=10, pad_sec=0.5):
        """
        Build chunks with pad_sec padding before/after each chunk, run ASR in batches,
        and map timestamps returned by ASR (assumed in ms relative to chunk start)
        back to absolute audio time.

        Parameters:
            diar_labels_local: [(st, ed, spk), ...]  # st/ed are seconds in full audio
            wav_cache_local: {seg_idx: np.ndarray(wav_seg)}  # wav_seg corresponds to [st:ed] of original
            sr_local: sample rate
            max_chunk_sec_local: target chunk length in seconds (without counting pad)
            overlap_sec: overlap between adjacent chunks (as before)
            batch_size: ASR batch size
            pad_sec: padding added to both chunk start and end, in seconds (default 0.5)
        Returns:
            output_segments, asr_results_batch, chunk_info
            - chunk_info entries are (seg_idx, chunk_start_time, spk)
              where chunk_start_time is the absolute time (seconds) of the START of the **padded** chunk
        """
        all_chunks = []
        chunk_info = []

        pad_samples = int(pad_sec * sr_local)
        max_len = int(max_chunk_sec_local * sr_local)
        step = max_len - int(overlap_sec * sr_local)

        # 1) 构建 chunk（带前后 pad）
        for seg_idx, (st, ed, spk) in enumerate(diar_labels_local):
            wav_seg = wav_cache_local.get(seg_idx)
            if wav_seg is None or len(wav_seg) == 0:
                continue

            total_len = len(wav_seg)
            start = 0
            while start < total_len:
                chunk_start = max(0, start - int(overlap_sec * sr_local))
                chunk_end = min(start + max_len, total_len)

                # apply padding (clip to segment bounds)
                padded_start = max(0, chunk_start - pad_samples)
                padded_end = min(total_len, chunk_end + pad_samples)

                all_chunks.append(wav_seg[padded_start:padded_end])

                # chunk_start_time should be absolute time of padded_start:
                # wav_seg corresponds to audio segment starting at `st` seconds,
                # so padded_start sample index maps to time st + padded_start / sr_local
                chunk_start_time = float(st + (padded_start / sr_local))
                chunk_info.append((seg_idx, chunk_start_time, spk))

                start += step

        # 2) 分批 ASR（保持原逻辑）
        asr_results_batch = []
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_results = asr_model(batch_chunks)
            asr_results_batch.extend(batch_results)

        # 3) 构建输出：把 ASR 中的 timestamp(ms) 映射到绝对时间（秒）
        output_segments = []
        for chunk_idx, asr_results in enumerate(asr_results_batch):
            seg_idx, chunk_start_time, spk = chunk_info[chunk_idx]
            if isinstance(asr_results, dict):
                asr_results = [asr_results]

            for sent in asr_results:
                ts_list = sent.get('timestamp', [])
                # ts_list 中元素假设为 (t0_ms, t1_ms)
                timestamps = [
                    [chunk_start_time + (t0_ms / 1000.0), chunk_start_time + (t1_ms / 1000.0)]
                    for t0_ms, t1_ms in ts_list
                ]
                preds = sent.get('preds', "")
                if preds:
                    output_segments.append({
                        "segment": seg_idx,
                        "speaker": spk,
                        "text": preds,
                        "timestamp": timestamps
                    })

        return output_segments, asr_results_batch, chunk_info

    t_asr = time.time()
    output_segments, asr_results_batch, chunk_info = run_asr(diar_labels, wav_cache)
    PROFILE_STATS.setdefault("", []).append(time.time() - t_asr)
    logger.info("ASR : %.3fs, raw_segments=%d", PROFILE_STATS.get("", [0.0])[-1], len(output_segments))

    # 后处理：去重、合并
    output_segments = deduplicate_tokens(output_segments, overlap_threshold=0.3)
    output_segments = merge_output_segments(output_segments, max_gap=1.5)

    # ITN
    t_itn = time.time()
    output_segments = add_tokens(output_segments)
    texts = [seg["text"] for seg in output_segments]
    texts_itn = itn_process_batch(texts, lang="zh")
    output_segments = update_tokens_with_itn(output_segments, texts_itn)
    PROFILE_STATS.setdefault("ITN", []).append(time.time() - t_itn)
    logger.info("ITN : %.3fs", PROFILE_STATS.get("ITN", [0.0])[-1])

    # 标点 + 语义纠错（两步保证 token/timestamp 对齐）
    t_punc = time.time()
    texts_for_corr = [seg["text"] for seg in output_segments if seg.get("text")]

    # 1) 批量加标点（punc_model 返回可能为 (text,score) 或 list）
    punc_texts = [punc_model(t)[0] if callable(punc_model) else t for t in texts_for_corr]
    punc_texts = [pt[0] if isinstance(pt, (list, tuple)) else pt for pt in punc_texts]

    # 先更新标点
    output_segments = update_tokens_with_corr_and_punc(
        output_segments,
        corrected_texts=texts_for_corr,
        punc_texts=punc_texts,
        debug=False
    )

    # 再语义纠错（基于标点后的文本）
    texts_after_punc = [seg["text"] for seg in output_segments if seg.get("text")]
    if sem_correction_model is not None:
        corrected_texts = sem_correction_model.batch_correct(texts_after_punc)
    else:
        corrected_texts = texts_after_punc

    output_segments = update_tokens_with_corr_and_punc(
        output_segments,
        corrected_texts=corrected_texts,
        punc_texts=corrected_texts,
        debug=False
    )
    PROFILE_STATS.setdefault("PUNC_AND_CORR", []).append(time.time() - t_punc)
    logger.info("PUNC & CORR : %.3fs", PROFILE_STATS.get("PUNC_AND_CORR", [0.0])[-1])

    total_elapsed = time.time() - total_start
    # 汇总 timings（从 PROFILE_STATS 取总和，兜底为 0.0）
    timings = {stage: sum(times) for stage, times in PROFILE_STATS.items()}
    timings.setdefault("Total", total_elapsed)
    logger.info("TOTAL : %.3fs", total_elapsed)

    return output_segments, timings


