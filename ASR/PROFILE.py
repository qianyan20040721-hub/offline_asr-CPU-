import time
import logging
import numpy as np
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 全局 profile 统计
PROFILE_STATS: Dict[str, List[float]] = {}

def profile_stage(stage_name: str):
    """装饰器：记录阶段耗时"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            PROFILE_STATS.setdefault(stage_name, []).append(elapsed)
            logger.info("[PROFILE] Stage=%s Time=%.3fs", stage_name, elapsed)
            return result
        return wrapper
    return decorator

def print_profile_summary(audio_len: float):
    total_time = sum(sum(times) for times in PROFILE_STATS.values())
    logger.info("\n[SUMMARY]")
    logger.info("    Audio Length = %.2fs", audio_len)
    for stage, times in PROFILE_STATS.items():
        avg = sum(times) / len(times)
        logger.info("    %-12s calls=%d total=%.2fs avg=%.3fs",
                    stage, len(times), sum(times), avg)
    logger.info("    Total        = %.2fs", total_time)
    logger.info("    Overall RT   = %.3fx", total_time / audio_len)