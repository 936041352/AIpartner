"""根据当前 Profile Facts 生成稳定的用户画像。

公开入口：update_user_profile(memory_folder, thinking=False, force=False)
正文和 source_snapshot 一起保存；快照每条只含 id、updated_at。

目录约定：
    <character_dir>/<memory_folder>/long_memory/profile_facts.json
    <character_dir>/<memory_folder>/long_memory/user_profile.json
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser

from ..character_setting import summary_character_setting
from ..llm import get_llm, get_thinking_llm
from .profile_fact_extractor import (
    FACTS_PATH,
    load_profile_fact_state,
)


PROFILE_PATH = Path("long_memory") / "user_profile.json"


class UserProfileResult(BaseModel):
    """用户画像模型输出。"""

    profile_summary: str = Field(min_length=1)


def _alert(message: str) -> None:
    """输出用户画像更新报警。

    Args:
        message: 需要报警的错误信息。

    Returns:
        无返回值。项目接入正式报警系统时，只需替换此函数。
    """
    print(f"[更新用户画像][ALERT] {message}")


def _response_text(response: Any) -> str:
    """从常见 LangChain 响应中提取文本。

    Args:
        response: ``llm.invoke`` 返回值。

    Returns:
        供 PydanticOutputParser 解析的字符串。
    """
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _save_json(path: Path, data: Dict[str, Any]) -> bool:
    """原子保存 JSON 文件。

    Args:
        path: 目标文件路径。
        data: 要保存的 JSON 对象。

    Returns:
        保存成功返回 True，失败时报警并返回 False。
    """
    temp_path = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
        return True
    except OSError as error:
        _alert(f"保存用户画像失败：{error}")
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _load_previous_state(path: Path) -> Dict[str, Any]:
    """读取上一版用户画像及其来源快照。

    Args:
        path: ``user_profile.json`` 路径。

    Returns:
        完整状态；缺失或无效时返回空字典，允许随后重新生成。
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"[更新用户画像] 上一版画像读取失败，将按首次生成处理：{error}")
        return {}
    return data if isinstance(data, dict) else {}


def _snapshot_map(snapshot: Any) -> Optional[Dict[str, float]]:
    """校验快照并转换为映射，忽略记录顺序。

    Args:
        snapshot: 包含 id、updated_at 的列表，也可传入完整事实列表。
    Returns:
        ID 到时间戳的映射；缺失、重复 ID 或字段无效时返回 None。
    """
    if not isinstance(snapshot, list):
        return None
    result = {}
    for item in snapshot:
        if not isinstance(item, dict):
            return None
        item_id, timestamp = item.get("id"), item.get("updated_at")
        if (
            not isinstance(item_id, str) or not item_id.strip()
            or item_id in result
            or not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool)
            or not math.isfinite(timestamp) or timestamp < 0
        ):
            return None
        result[item_id] = timestamp
    return result


def _format_facts(
    facts: list[Dict[str, Any]], previous_snapshot: Dict[str, float],
) -> str:
    """为事实标记变化，先展示有变化项，再展示未变化项。

    Args:
        facts: 当前全部事实。
        previous_snapshot: 上次成功总结的 ID/updated_at 映射；无有效快照时为空。
    Returns:
        逐行格式化文本；各组内部保留输入顺序，不修改事实或时间戳。
    """
    changed, unchanged = [], []
    for fact in facts:
        same = (
            fact["id"] in previous_snapshot
            and fact["updated_at"] == previous_snapshot[fact["id"]]
        )
        mark = "未变化" if same else "有变化"
        line = (
            f"[{fact['id']} | {mark} | category={fact['category']} | "
            f"importance={fact['importance']}] {fact['content']}"
        )
        (unchanged if same else changed).append(line)
    return "\n".join(changed + unchanged) or "（暂无）"


def _build_profile_prompt(
    facts: list[Dict[str, Any]],
    previous_profile: str,
    character_setting_summary: str,
    parser: PydanticOutputParser,
    retry_note: str = "",
    previous_snapshot: Optional[Dict[str, float]] = None,
) -> str:
    """构建用户画像提示词。

    Args:
        facts: 当前全部 Profile Facts。
        previous_profile: 上一版用户画像，可为空。
        character_setting_summary: 当前角色设定总结。
        parser: 用户画像 PydanticOutputParser。
        retry_note: 第二次尝试时附加的失败说明。
        previous_snapshot: 上次成功总结使用的 ID/updated_at 映射。

    Returns:
        可直接传给 ``llm.invoke`` 的提示词。
    """
    previous_text = previous_profile or "（暂无，这是第一次生成）"
    retry_text = f"\n【上次失败信息】\n{retry_note}\n" if retry_note else ""
    # 加载函数只读取事实；本模块根据旧快照独立标记变化并调整展示顺序。
    facts_text = _format_facts(facts, previous_snapshot or {})
    return f"""你是 AI 陪伴系统中的用户画像整理器。

你的输出只负责 Profile。Core Episode 和 Active Thread 将由其他模块维护，
不得把三类记忆混写成一份笼统的长期记忆。

【信息优先级】
1. 当前 Profile Facts 是判断事实真伪的唯一权威来源，但并非其中每条内容
   都适合进入用户画像；你必须先筛选类型，再进行总结。
2. 上一版用户画像只用于保持合格 Profile 内容的措辞、结构和重点稳定。
3. 角色设定只用于理解当前 ASSISTANT 及双方关系语境。

如果三者不一致，必须以当前 Profile Facts 为准。

【当前 ASSISTANT 角色设定总结】
{character_setting_summary}

角色设定不是用户事实来源。不得把角色的身份、经历、性格、能力、世界观，
或角色单方面表达的关系写进用户画像。

【Profile Facts 格式】
每行格式为：[事实 ID | 有变化/未变化 | category=分类 | importance=重要性] 事实内容
只有 ID 存在于上次成功总结的快照且 updated_at 相同，才标为未变化；
其他一律标为有变化，无有效旧快照时全部视为有变化。
有变化项排在前面，未变化项排在后面；变化标记只帮助定位修改，不代表
事实更重要或刚刚发生。两组都必须结合理解，最终仍按画像主题自然组织。
ID、变化标签、category 和 importance 不得出现在最终画像中。

【当前 Profile Facts】
{facts_text}

【上一版用户画像】
{previous_text}

【三类长期记忆的边界】
一、Profile：本次写入目标
- 用户相对稳定的身份与长期背景。
- 长期兴趣、能力与经验。
- 稳定偏好、习惯、价值取向、思考或沟通方式。
- 用户对当前 ASSISTANT 的稳定关系定位、长期态度和互动偏好。
- 能跨越许多次对话，持续帮助 ASSISTANT 理解和回应用户的信息。

二、Core Episode：本次禁止写入
- 某次具体经历、阶段性成果、重要互动或有时间边界的故事。
- 生病、康复、争执、安慰、庆祝、共同完成某件事等事件经过或结果。
- 即使事件很重要、很感人或有长期纪念意义，也不应写进 Profile；
  它应由 Core Episode 模块保存。

三、Active Thread：本次禁止写入
- 当前正在进行、尚未完成或以后需要继续跟进的事项。
- 最近状态、短期安排、计划、目标、待办、开发进度、学习计划、期限。
- 暂时性的作息、身体状况、工作压力、课程或项目阶段。
- 它们应由 Active Thread 模块保存；完成后也不能自动转写成 Profile。

【筛选方法】
在写作前，先在内部逐条判断事实属于 PROFILE、EPISODE、THREAD 还是 DROP，
但不要输出判断过程：
- 只有 PROFILE 可以进入最终画像。
- 一个事实同时含有稳定特征和事件细节时，只保留可独立成立的稳定特征，
  删除人物、时间、进度、病情、任务和故事经过等事件性细节。
- “用户正在做某事”不能直接改写成“用户长期热爱某事”；只有 Facts 本身
  明确支持长期兴趣、能力或偏好时，才能保守概括。
- 某次接受安慰、提醒或建议，不足以证明稳定互动偏好；只有明确表达或
  多次一致证据支持时才可写入。
- 角色设定、系统运行方式、当前对话的必然条件，以及对理解用户无帮助的
  信息归为 DROP。
- 无法确定是否足够稳定时，宁可省略，不要放入画像兜底。

【边界示例】
以下各例彼此独立、均为虚构，不构成同一用户的画像，不得复制为实际记忆。
- “用户是博物馆讲解员，主要负责自然历史展区”可以作为阶段较长的身份背景。
- “用户本周因新展布置，每天提前到馆核对展品标签”属于 Active Thread，不写。
- “用户会使用缝纫机，能够独立修改衣物”可以作为能力背景；
  “用户准备下个月报名学习木工”属于 Active Thread，不写。
- “用户明确表示，讨论选择时，希望伙伴先列出选项，再给建议”可以作为互动偏好；
  “用户计划周末与伙伴一起挑选一盏台灯”属于 Active Thread，不写。
- “用户第一次独立带团前感到紧张，请伙伴陪自己练习开场白；活动现已结束”
  属于 Core Episode，不写；不能据此推断用户长期胆小或依赖陪伴。

【生成要求】
1. 只能依据当前 Profile Facts 中被判定为 PROFILE 的内容描述用户。
2. 上一版画像不是事实来源，也不是必须保留的内容清单。即使其中某句话
   仍能在 Facts 中找到依据，只要它属于 Episode、Thread 或 Drop，也必须删除。
3. 在通过类型筛选后，尽量延续上一版合格内容的措辞、段落结构和重点，
   只做必要修改，不要为了改写而改写。
4. 新的 Profile 信息应自然合并到相关内容，不要机械追加或逐条复述。
5. 保守归纳，不要过度补充或者推测，不得添加没有事实依据的信息，也不得通过改变措辞把
   临时状态或具体事件伪装成稳定特征。
6. 有充分事实支持时，应描述用户与当前 ASSISTANT 的稳定关系和互动偏好；
   不写具体互动事件，不复述角色设定。
7. 不得把兴趣、计划或单次行为强化为人格结论。
8. 最终画像应主要回答“用户长期是谁、在意什么、擅长什么、偏好怎样互动”，
   而不是“用户最近发生了什么、正在做什么、以后准备做什么”。
9. 不使用“最近、目前正在、刚刚、预计、计划、已经完成”等事件或进度式
   表达；描述阶段性身份时除外，例如“用户是博物馆讲解员”。
10. 使用第三人称、客观、自然、紧凑的中文表述。
11. 事实充分时控制在约 300～500 个中文字符；事实较少时可以更短，
    不得为了长度补充内容。
12. 不输出标题、内部字段、分类结果、生成过程或解释。
{retry_text}
【输出格式】
{parser.get_format_instructions()}

只返回符合 schema 的 JSON，不要附加 Markdown 代码块。
"""


def _generate_profile(
    llm: Any,
    facts: list[Dict[str, Any]],
    previous_profile: str,
    character_setting_summary: str,
    previous_snapshot: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """调用模型生成用户画像，失败时重试一次。

    Args:
        llm: 普通或思考模型。
        facts: 当前全部 Profile Facts。
        previous_profile: 上一版用户画像。
        character_setting_summary: 当前角色设定总结。
        previous_snapshot: 上次成功总结的快照映射，重试时保持不变。

    Returns:
        成功时返回画像正文；连续两次失败时报警并返回 None。
    """
    parser = PydanticOutputParser(pydantic_object=UserProfileResult)
    retry_note = ""

    for attempt in (1, 2):
        prompt = _build_profile_prompt(
            facts,
            previous_profile,
            character_setting_summary,
            parser,
            retry_note,
            previous_snapshot=previous_snapshot,
        )
        try:
            raw_output = _response_text(llm.invoke(prompt))
            result = parser.parse(raw_output)
            summary = result.profile_summary.strip()
            if summary:
                return summary
            retry_note = "profile_summary 为空，请重新生成。"
        except Exception as error:
            retry_note = f"第 {attempt} 次调用或解析失败：{error}"
            print(f"[更新用户画像] {retry_note}")

    _alert("用户画像连续两次生成失败，已保留上一版画像。")
    return None


def _update_profile_impl(
    memory_path: Path,
    character_setting_summary: str,
    thinking: bool,
    force: bool = False,
) -> Optional[str]:
    """读取当前 Facts 并生成用户画像，只保存画像文件。

    Args:
        memory_path: 当前对话使用的角色记忆文件夹。
        character_setting_summary: 角色设定总结。
        thinking: 是否使用思考模型。
        force: True 时忽略快照一致性并重新总结；空事实仍不调用模型。

    Returns:
        成功或无需更新时返回正文；空事实返回空字符串；失败返回 None。
    """
    facts_path = memory_path / FACTS_PATH
    state, existed, _ = load_profile_fact_state(facts_path, strict=False)
    if state is None or not existed:
        _alert(f"无法读取 Profile Facts：{facts_path}")
        return None
    profile_path = memory_path / PROFILE_PATH
    previous = _load_previous_state(profile_path)
    old_text = previous.get("profile_summary")
    previous_profile = old_text.strip() if isinstance(old_text, str) else ""
    current_map = _snapshot_map(state["facts"])
    if current_map is None:
        _alert("当前事实无法生成有效快照。")
        return None
    snapshot = [{"id": key, "updated_at": value} for key, value in current_map.items()]
    valid_text = isinstance(old_text, str) and (
        bool(previous_profile) if state["facts"] else not previous_profile
    )
    if not force and valid_text and _snapshot_map(previous.get("source_snapshot")) == current_map:
        return previous_profile

    if state["facts"]:
        try:
            llm = get_thinking_llm() if thinking else get_llm()
        except Exception as error:
            _alert(f"获取用户画像模型失败：{error}")
            return None
        final_profile = _generate_profile(
            llm, state["facts"], previous_profile, character_setting_summary,
            previous_snapshot=_snapshot_map(previous.get("source_snapshot")),
        )
        if final_profile is None:
            return None
    else:
        # 只有已成功读取的空 facts 才清空；读取失败不会走到这里。
        final_profile = ""

    profile_data = {
        "schema_version": 1,
        "profile_summary": final_profile,
        "updated_at": time.time(),
        "source_snapshot": snapshot,
    }
    if not _save_json(profile_path, profile_data):
        return None
    return final_profile


def update_user_profile(
    memory_folder: str | Path,
    thinking: bool = False,
    force: bool = False,
) -> Optional[str]:
    """根据最新 Facts 更新用户画像。

    Args:
        memory_folder: 当前对话实际使用的角色记忆文件夹。
        thinking: True 使用思考模型，False 使用普通模型。
        force: True 强制重新总结；False 时快照一致且正文有效便直接返回。

    Returns:
        成功或无需更新时返回正文，明确空事实返回空字符串。
        普通失败打印日志并返回 None，原文件及快照保留。
        本模块不决定是否阻止聊天，由整合层检查最终落盘状态。

    Raises:
        Exception: 角色设定总结读取或生成失败时，原异常继续向上抛出。
        RuntimeError: 角色设定总结为空时抛出。
    """
    print(f"[更新用户画像] 正在根据用户事实更新用户画像...")
    try:
        memory_path = Path(memory_folder).expanduser()
        character_dir = memory_path.parent
        character_name = character_dir.name
    except Exception as error:
        _alert(f"解析角色目录失败：{error}")
        return None

    # 角色设定是必要上下文，故意放在通用异常兜底之外。
    character_setting_summary = summary_character_setting(
        character_dir,
        character_name,
    )
    if (
        not isinstance(character_setting_summary, str)
        or not character_setting_summary.strip()
    ):
        raise RuntimeError(f"角色 {character_name} 的角色设定总结为空。")

    try:
        return _update_profile_impl(
            memory_path,
            character_setting_summary.strip(),
            thinking,
            force,
        )
    except Exception as error:
        _alert(f"用户画像更新出现未预期异常：{error}")
        return None
