"""
本地语音识别（ASR）模块。

使用 faster-whisper 加载本地 CTranslate2 格式的 Whisper 模型，为网页端
"麦克风语音输入"提供转写能力。

设计要点：

- 模型按需惰性加载，整个进程只加载一次；首次加载由角色初始化时的预热
  提前完成，避免用户点击麦克风时长时间等待。
- 默认使用 CPU + int8：显卡需要留给 Qwen3-TTS，显存不足时不与语音合成
  争用；同时避免 CUDA 运行时（cuBLAS/cuDNN）缺失导致无法使用。
- 模型权重默认读取 .env 中的 ASR_MODEL_DIR，未配置时尝试项目内
  weights 目录以及 HuggingFace 缓存目录。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# HuggingFace 缓存目录命名（faster-whisper 与 transformers 的 cache 布局不同）
_HF_CACHE_DIR_NAMES = (
    "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
    "models--Systran--faster-whisper-large-v3",
    "models--Systran--faster-whisper-medium",
)

# CTranslate2 模型目录必须包含这些文件
_REQUIRED_MODEL_FILES = ("model.bin", "config.json")


class ASRUnavailableError(RuntimeError):
    """语音识别不可用（缺少依赖、模型或初始化失败）。"""


class ASRTranscribeError(RuntimeError):
    """语音识别过程失败。"""


_model_lock = threading.Lock()
_model: Any = None
_model_dir: Path | None = None
_model_error: str | None = None


def _normalize_config_path(raw: str) -> str:
    """
    修正 .env 中 Windows 路径被转义字符破坏的情况。

    python-dotenv 对双引号内的反斜杠会做转义处理，例如
    "F:\\Program Files\\voice_program" 中的 ``\\v`` 会被解析成垂直制表符，
    导致路径变成 ``F:\\Program Files\\x0boice_program``。真实路径里几乎
    不可能出现控制字符，这里把它们还原成对应的字母。
    """
    replacements = {"\x0b": "v", "\x0c": "f", "\x07": "a", "\x08": "b"}
    result = raw
    for broken, letter in replacements.items():
        if broken in result:
            result = result.replace(broken, letter)
    return result


def _candidate_model_dirs() -> list[Path]:
    """按优先级返回候选模型目录。"""
    candidates: list[Path] = []

    configured = (os.getenv("ASR_MODEL_DIR") or "").strip()
    if configured:
        candidates.append(Path(_normalize_config_path(configured)).expanduser())

    # 项目内权重目录（可选）
    candidates.append(
        _PROJECT_ROOT
        / "weights"
        / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
    )

    # HuggingFace 缓存目录
    hf_home = (os.getenv("HF_HOME") or "").strip()
    cache_roots: list[Path] = []
    if hf_home:
        cache_roots.append(Path(hf_home) / "hub")
    hf_hub_cache = (os.getenv("HUGGINGFACE_HUB_CACHE") or "").strip()
    if hf_hub_cache:
        cache_roots.append(Path(hf_hub_cache))
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    for root in cache_roots:
        for name in _HF_CACHE_DIR_NAMES:
            candidates.append(root / name)

    return candidates


def _is_valid_model_dir(path: Path) -> bool:
    """判断目录是否为可用的 CTranslate2 模型目录。"""
    if not path.is_dir():
        return False
    return all((path / name).is_file() for name in _REQUIRED_MODEL_FILES)


def resolve_model_dir() -> Path | None:
    """返回第一个可用的模型目录；找不到时返回 None。"""
    for candidate in _candidate_model_dirs():
        if _is_valid_model_dir(candidate):
            return candidate

    # 兼容 HuggingFace 缓存中的 snapshots/<hash>/ 布局
    for candidate in _candidate_model_dirs():
        snapshots = candidate / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in sorted(snapshots.iterdir(), reverse=True):
            if _is_valid_model_dir(snapshot):
                return snapshot
    return None


def _build_model(model_dir: Path):
    """构造 WhisperModel；依赖缺失时抛出 ASRUnavailableError。"""
    try:
        from faster_whisper import WhisperModel
    except Exception as error:  # pragma: no cover - 依赖缺失路径
        raise ASRUnavailableError(
            "未安装 faster-whisper，无法使用语音输入。"
            "请执行：python -m pip install faster-whisper"
        ) from error

    device = (os.getenv("ASR_DEVICE") or "cpu").strip().lower()
    default_compute = "int8" if device == "cpu" else "float16"
    compute_type = (os.getenv("ASR_COMPUTE_TYPE") or default_compute).strip()

    try:
        cpu_threads = int((os.getenv("ASR_CPU_THREADS") or "").strip() or 0)
    except ValueError:
        cpu_threads = 0
    if cpu_threads <= 0:
        cpu_threads = min(os.cpu_count() or 8, 16)

    return WhisperModel(
        str(model_dir),
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )


def get_model():
    """
    返回已加载的共享模型实例。

    首次调用会加载模型；并发调用会串行等待同一次加载。
    """
    global _model, _model_dir, _model_error

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        model_dir = resolve_model_dir()
        if model_dir is None:
            attempted = "；".join(str(p) for p in _candidate_model_dirs())
            raise ASRUnavailableError(
                "未找到可用的语音识别模型。请设置 .env 中的 ASR_MODEL_DIR "
                f"指向 faster-whisper 模型目录。已尝试：{attempted}"
            )

        print(f"[语音识别] 正在加载模型：{model_dir}")
        try:
            _model = _build_model(model_dir)
        except ASRUnavailableError:
            raise
        except Exception as error:
            _model_error = f"{type(error).__name__}: {error}"
            raise ASRUnavailableError(
                f"语音识别模型加载失败：{_model_error}"
            ) from error

        _model_dir = model_dir
        print("[语音识别] 模型加载完成。")
        return _model


def warmup() -> bool:
    """
    预热模型，供角色初始化时后台调用。

    失败不抛异常，只返回是否成功，避免影响正常对话。
    """
    try:
        model = get_model()
    except Exception as error:
        print(f"[语音识别] 预热失败：{type(error).__name__}: {error}")
        return False

    # 用极短静音跑一次，确保解码路径与 VAD 都已就绪。
    try:
        import numpy as np

        silence = np.zeros(8000, dtype=np.float32)
        list(model.transcribe(silence, language="zh", beam_size=1)[0])
    except Exception as error:
        print(f"[语音识别] 预热推理失败：{type(error).__name__}: {error}")
        return False
    return True


def is_ready() -> bool:
    """模型是否已经加载完成。"""
    return _model is not None


def describe() -> dict[str, Any]:
    """返回当前 ASR 配置与状态，供前端判断是否显示麦克风。"""
    model_dir = _model_dir or resolve_model_dir()
    return {
        "available": model_dir is not None,
        "loaded": _model is not None,
        "model_dir": str(model_dir) if model_dir else None,
        "device": (os.getenv("ASR_DEVICE") or "cpu").strip().lower(),
        "error": _model_error,
    }


def transcribe(
    audio_path: str | Path,
    *,
    language: str | None = None,
    beam_size: int | None = None,
) -> dict[str, Any]:
    """
    转写一个音频文件。

    :param audio_path: 待转写音频路径（wav/mp3/m4a 等，由 PyAV 解码）。
    :param language: 语言代码；留空时读取 ASR_LANGUAGE（缺省 zh）。
        显式传 "auto" 可强制自动检测，但会多一次编码器前向、明显变慢。
    :param beam_size: 束搜索宽度；默认读取 ASR_BEAM_SIZE，缺省 5。
    :return: {"text", "language", "language_probability", "duration"}
    """
    model = get_model()

    if beam_size is None:
        try:
            beam_size = int((os.getenv("ASR_BEAM_SIZE") or "").strip() or 5)
        except ValueError:
            beam_size = 5

    if language is None:
        # 已知语言可以跳过自动检测：省掉一次完整编码器前向，实测约省一半耗时。
        # 需要自动判别时在 .env 里设置 ASR_LANGUAGE=""（空值）或传 "auto"。
        configured = os.getenv("ASR_LANGUAGE")
        language = "zh" if configured is None else configured.strip()
    if language.lower() in {"", "auto", "none"}:
        language = None

    try:
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=max(1, beam_size),
            vad_filter=True,
        )
        text = "".join(segment.text for segment in segments).strip()
    except Exception as error:
        raise ASRTranscribeError(
            f"语音识别失败：{type(error).__name__}: {error}"
        ) from error

    return {
        "text": text,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
    }
