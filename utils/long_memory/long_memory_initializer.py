"""更新并检查长期记忆，检查通过后才返回可用于聊天的文本。

同一 memory_folder 的更新必须串行执行。
整合层不重试模型、不迁移旧记忆、不写额外状态文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .profile_fact_extractor import (
    FACTS_PATH, HISTORY_PATH, CATEGORIES, extract_profile_facts,
    _load_turns as _load_profile_turns,
)
from .core_episode_extractor import (
    EPISODES_PATH, EPISODE_STATUSES, extract_core_episodes,
    _load_turns as _load_episode_turns,
    _normalize_episode,
)
from .user_profile_summarizer import (
    PROFILE_PATH, update_user_profile, _snapshot_map,
)
from .core_journey_summarizer import JOURNEY_PATH, summarize_core_journey
from .episode_buffer_sync import sync_episode_buffer


MAX_UNPROCESSED_TURNS = 100
MAX_PROFILE_SNAPSHOT_DIFF = 6
MAX_JOURNEY_SNAPSHOT_DIFF = 2
VALID_SCENE_MODES = ("realtime", "sandbox")


def _read_json(path: Path, errors: List[str]) -> Optional[Dict[str, Any]]:
    """读取必须存在的状态文件，不把读取失败当作空状态。

    Args:
        path: 输入 JSON 文件路径。
        errors: 本次检查错误列表，失败信息追加到此列表。
    Returns:
        顶层对象；读取或格式错误返回 None，由最终检查统一抛出异常。
    """
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            errors.append(f"{path.name} 顶层必须是对象。")
            return None
        return data
    except (OSError, ValueError) as error:
        errors.append(f"{path.name} 无法读取：{error}")
        return None


def _check_backlog(
    label: str, state: Optional[Dict[str, Any]],
    turns: Optional[List[Dict[str, Any]]], errors: List[str],
) -> None:
    """按提取器的有效 turn 列表检查积压，不对 ID 做减法。

    Args:
        label: 用于错误说明的链路名。
        state: 已落盘的提取状态，读取失败时为 None。
        turns: 该提取器清洗后的历史，读取失败时为 None。
        errors: 收集错误的列表。
    Returns:
        无返回值；超过100轮、历史无效或游标无法定位时追加错误。
    """
    if turns is None:
        errors.append(f"{label} 历史读取失败，无法检查积压。")
        return
    if state is None:
        return
    if "last_processed_turn_id" not in state:
        errors.append(f"{label} 缺少 last_processed_turn_id。")
        return
    cursor = state["last_processed_turn_id"]
    if cursor is None:
        pending = len(turns)
    else:
        position = next((i for i, turn in enumerate(turns) if turn["turn_id"] == cursor), None)
        if not isinstance(cursor, str) or not cursor or position is None:
            errors.append(f"{label} 处理游标无法在保留历史中定位：{cursor!r}。")
            return
        pending = len(turns) - position - 1
    if pending > MAX_UNPROCESSED_TURNS:
        errors.append(f"{label} 积压 {pending} 轮，允许最多 {MAX_UNPROCESSED_TURNS} 轮。")


def _check_summary(
    label: str, source: Optional[Dict[str, Any]], source_key: str,
    summary: Optional[Dict[str, Any]], text_key: str,
    limit: int, errors: List[str],
) -> str:
    """检查源记录、总结正文和来源快照，并计算不同 ID 的总数。

    Args:
        label: 用户画像或核心经历的显示名。
        source: 已保存的源数据对象。
        source_key: facts 或 core_episodes；不会比较 Episode 缓存。
        summary: 已保存的总结对象。
        text_key: profile_summary 或 core_journey。
        limit: 允许的快照差异数，只有严格超过才报错。
        errors: 错误信息追加位置。
    Returns:
        可用于组装的正文；无效时返回空串并记录错误，最终统一阻断。
    """
    if source is None or summary is None:
        return ""
    items = source.get(source_key)
    current = _snapshot_map(items)
    previous = _snapshot_map(summary.get("source_snapshot"))
    if current is None:
        errors.append(f"{label} 源记录 ID 或 updated_at 无效、缺失或重复。")
        return ""
    # 基础正文校验；不重新排序、截断或修复任何源文件。
    required = ("content", "category") if source_key == "facts" else ("content", "title", "status")
    if any(
        any(not isinstance(item.get(key), str) or not item[key].strip() for key in required)
        or not isinstance(item.get("importance"), int)
        or isinstance(item.get("importance"), bool)
        or not 1 <= item["importance"] <= 5
        or (item["category"] not in CATEGORIES if source_key == "facts"
            else item["status"] not in EPISODE_STATUSES)
        for item in items
    ):
        errors.append(f"{label} 源记录缺少有效正文、分类／状态或重要度。")
    text = summary.get(text_key)
    if not isinstance(text, str):
        errors.append(f"{label} 总结正文格式无效。")
        text = ""
    text = text.strip()
    if previous is None:
        errors.append(f"{label} 来源快照缺失或无效，需要重新总结。")
    if not current:
        # 空源必须明确清空；不能用差异阈值保留已经全部删除的记忆。
        if text or previous != {}:
            errors.append(f"{label} 源数据已空，但正文或快照尚未清空。")
        return ""
    if not text:
        errors.append(f"{label} 源数据非空，却没有可用总结。")
    if previous is not None:
        # 并集内每个 ID 仅计一次：新增、退出、时间变化都算1条。
        diff = sum(
            item_id not in current or item_id not in previous
            or current[item_id] != previous[item_id]
            for item_id in current.keys() | previous.keys()
        )
        if diff > limit:
            errors.append(f"{label} 快照差异 {diff} 条，允许最多 {limit} 条。")
    return text


def _check_and_load(memory_path: Path, scene_mode: str) -> Tuple[str, str]:
    """读取最终状态，汇总检查问题，通过后返回两份正文。

    Args:
        memory_path: 当前角色记忆目录。
        scene_mode: realtime 或 sandbox。
    Returns:
        (用户画像, 核心经历)，合法空记忆对应空字符串。
    Raises:
        RuntimeError: 格式无效、模式不符、积压或快照差异超限。
    """
    errors: List[str] = []
    facts = _read_json(memory_path / FACTS_PATH, errors)
    episodes = _read_json(memory_path / EPISODES_PATH, errors)
    profile = _read_json(memory_path / PROFILE_PATH, errors)
    journey = _read_json(memory_path / JOURNEY_PATH, errors)
    if episodes is not None:
        # 缓存不参与快照差异计算，但文件损坏不能被当作健康状态。
        core, buffer = episodes.get("core_episodes"), episodes.get("episode_buffer")
        if not isinstance(core, list) or not isinstance(buffer, list):
            errors.append("事件文件的 core_episodes、episode_buffer 必须是数组。")
        elif _snapshot_map(core + buffer) is None:
            errors.append("事件文件存在无效记录时间戳或跨区域重复 ID。")
        elif any(_normalize_episode(item, 0.0) is None for item in buffer):
            errors.append("事件缓存存在字段无效的记录。")
    # 分别复用两个提取器的清洗规则，不改变“有效轮数”的定义。
    profile_turns = _load_profile_turns(memory_path / HISTORY_PATH)
    episode_turns = _load_episode_turns(memory_path / HISTORY_PATH, scene_mode)
    _check_backlog("Profile Facts", facts, profile_turns, errors)
    _check_backlog("Core Episodes", episodes, episode_turns, errors)
    profile_text = _check_summary(
        "用户画像", facts, "facts", profile, "profile_summary",
        MAX_PROFILE_SNAPSHOT_DIFF, errors,
    )
    journey_text = _check_summary(
        "核心经历", episodes, "core_episodes", journey, "core_journey",
        MAX_JOURNEY_SNAPSHOT_DIFF, errors,
    )
    if journey is not None and journey.get("scene_mode") != scene_mode:
        errors.append("核心经历总结模式与当前对话不一致，需要重新总结。")
    if errors:
        raise RuntimeError("长期记忆检查失败，当前对话前提不满足：\n- " + "\n- ".join(errors))
    return profile_text, journey_text


def build_long_memory_context(
    memory_folder: str | Path, scene_mode: str = "realtime",
) -> str:
    """只读检查并组装聊天上下文；不调用模型、不修改文件。

    Args:
        memory_folder: 当前角色记忆目录。
        scene_mode: 当前对话模式，不通过文件夹名称推断。
    Returns:
        带“用户画像”“重要经历”标题的正文，空部分省略，两者皆空返回空串。
    Raises:
        RuntimeError: 目录／模式无效或任一长期记忆检查失败。
    """
    if scene_mode not in VALID_SCENE_MODES:
        raise RuntimeError(f"非法 scene_mode：{scene_mode!r}")
    try:
        memory_path = Path(memory_folder).expanduser()
    except (TypeError, ValueError, OSError) as error:
        raise RuntimeError(f"记忆目录无效：{error}") from error
    profile, journey = _check_and_load(memory_path, scene_mode)
    parts = []
    if profile:
        parts.append(f"【对用户的了解】\n{profile}")
    if journey:
        parts.append(f"【重要共同经历】\n{journey}")
    return "\n\n".join(parts)


def generate_long_memory(
    memory_folder: str | Path,
    thinking: bool = False,
    strict: bool = False,
    force: bool = True,
    scene_mode: str = "realtime",
    timezone_name: str = "Asia/Shanghai",
    force_summary: bool = False,
    *,
    memory_manager: Any = None,
    character_name: str = "",
) -> str:
    """先更新两条记忆链路，再检查落盘状态；通过后才返回聊天上下文。

    Args:
        memory_folder: 当前角色记忆目录，首次使用也须有合法 history.json。
        thinking: 传给四个模块，选择思考或普通模型。
        strict: 传给两个提取器，控制已有源数据的读取策略。
        force: 是否处理不足50轮的尾部对话，不强制重写总结。
        scene_mode: 传给事件提取和 Journey 总结，不传给 Profile 模块。
        timezone_name: 实时事件提取使用的时区；sandbox 忽略。
        force_summary: 强制重写两份总结，可用于修改提示词后重新生成。
        memory_manager: 已初始化的角色数据库；不传时保持原有的纯文本生成行为。
        character_name: 同步缓存使用的角色标识；同步失败只打印，不影响返回。
    Returns:
        检查通过的长期记忆文本；两份总结合法为空时返回空字符串。
    Raises:
        RuntimeError: 最终状态超出滞后阈值、格式／配置不合法。
        Exception: 必要角色设定总结异常原样传播；不包裹通用异常兜底。
    """
    if scene_mode not in VALID_SCENE_MODES:
        raise RuntimeError(f"非法 scene_mode：{scene_mode!r}")
    # 子模块负责各自的调用重试与普通错误处理；返回值不代表整体已同步。
    # 只根据最终文件判断是否允许聊天，部分批次失败不会回滚成功批次。
    if scene_mode == "realtime":
        try:
            ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError, TypeError) as error:
            raise RuntimeError(f"实时模式时区不可用：{timezone_name!r}") from error
    # 整理用户画像事实
    extract_profile_facts(memory_folder, thinking=thinking, strict=strict, force=force)
    # 更新用户画像。为了准确性，强制思考
    update_user_profile(memory_folder, thinking=True, force=force_summary)
    # 整理核心事件
    extract_core_episodes(
        memory_folder, thinking=thinking, strict=strict, force=force,
        scene_mode=scene_mode, timezone_name=timezone_name,
    )
    # 更新核心经历。为了准确性，强制思考
    summarize_core_journey(
        memory_folder, thinking=True, scene_mode=scene_mode, force=force_summary,
    )
    context = build_long_memory_context(memory_folder, scene_mode=scene_mode)
    # 最终检查通过后再同步辅助检索副本，保持只读上下文入口不变。
    if memory_manager is not None:
        sync_episode_buffer(memory_folder, memory_manager, character_name)
    return context
