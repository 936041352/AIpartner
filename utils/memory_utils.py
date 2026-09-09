from typing import Set, Iterable, List, Dict, Tuple, Any, Literal
import time
import json
import re
from datetime import datetime
from pydantic import BaseModel, Field
from pathlib import Path
from collections import deque
from collections.abc import Iterable, Mapping
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import RemoveMessage
from .character_config import VALID_SCENE_MODES
from .prompt_container import RECENT_MEMORY_PROMPTS, ARCHIVE_MEMORY_PROMPTS
from .display import CharacterDisplayService
from .long_memory.long_memory_initializer import generate_long_memory


# 结构化输出定义 (Pydantic Models)
class SingleMemory(BaseModel):
    text: str = Field(
        description="提炼出的核心记忆文本。必须使用第三人称陈述句，精简准确。"
    )
    category: Literal[
        "user_fact",
        "character_setting",
        "event",
    ] = Field(
        description="分类标签。"
        "'user_fact': 记录 User 明确表达的事实、当前状态、动作、情绪、经历、计划、偏好或观点。"
        "'character_setting': 记录 AI伙伴的事实、当前状态、动作、情绪、经历、计划、偏好或观点。"
        "'event': 记录双方已经发生或明确确认的互动事件。"
    )
    importance: int = Field(
        ge=1,
        le=10,
        description="重要度评级(1-10)。"
        "1-4分: 短期日常信息，用于记录即时状态、普通行为、临时情绪和生活流水账。"
        "5-7分: 持续性背景或明确偏好，用于记录具有一定持续性、会影响后续对话的背景信息、明确喜好、近期计划、重要约定或者亲密互动。"
        "8-10分: 核心且长期重要的信息，用于记录核心身份、重大经历、明确禁忌、长期目标，或对双方关系有重大影响的承诺和事件。"
    )


class MemoryExtraction(BaseModel):
    memories: list[SingleMemory] = Field(
        description=(
            "提取出的记忆列表。"
            "当前待总结对话中只要存在有效新增内容，"
            "就必须使用user_fact、character_setting或event中的适当类型予以覆盖。"
            "如果没有有效新增内容，必须严格返回空列表[]。"
        )
    )


def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_chat_history(
    unsaved_messages: Iterable[BaseMessage],
    character_name: str,
    ignored_tool_names: Set[str] = None
) -> tuple[str, list[BaseMessage]]:
    """
    将未保存的消息列表格式化为纯文本字符串，供大模型进行记忆提取反思。
    具备工具过滤机制：自动忽略黑名单中工具的调用记录及返回结果。

    :param unsaved_messages: 暂存的对话消息列表
    :param character_name: 当前 AI 角色的名称
    :param ignored_tool_names: 需要忽略的工具名称集合（默认忽略记忆检索和网络搜索）
    :return: 格式化后的对话历史字符串
    """
    if ignored_tool_names is None:
        ignored_tool_names = {
            "search_memory_tool",
            "web_search_tool",
        }

    chat_history = ""
    ignored_tool_call_ids = set()
    filtered_messages = []

    for msg in unsaved_messages:
        # 获取消息纯文本内容
        content = msg.content if hasattr(
            msg, 'content') else getattr(msg, 'text', '')

        # 1. 处理 AI 消息
        if isinstance(msg, AIMessage):
            # 记录黑名单里的 tool_call_id
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") in ignored_tool_names:
                        ignored_tool_call_ids.add(tc.get("id"))

            # 拼接入对话
            if content and isinstance(content, str) and content.strip():
                chat_history += f"{character_name}: {content.strip()}\n"

        # 2. 处理 工具返回 消息
        elif isinstance(msg, ToolMessage):
            # 跳过黑名单中的工具结果
            if msg.tool_call_id in ignored_tool_call_ids:
                filtered_messages.append(msg)
                continue

            # 保留其他外部工具的客观上下文
            tool_name = getattr(msg, 'name', None) or "未知工具"
            if content and isinstance(content, str) and content.strip():
                chat_history += f"[系统工具 {tool_name} 返回信息]: {content.strip()}\n"

        # 3. 处理 用户 消息
        elif isinstance(msg, HumanMessage):
            if content and isinstance(content, str) and content.strip():
                chat_history += f"User: {content.strip()}\n"

    return chat_history, filtered_messages


def format_chat_history_only_human_ai(
    unsaved_messages: Iterable[BaseMessage],
    character_name: str,
    display_service: CharacterDisplayService | None = None
) -> tuple[str, list[BaseMessage]]:
    """
    将未保存的消息列表格式化为纯文本字符串，供大模型进行记忆提取反思。
    只格式化AIMessage和HumanMessage

    :param unsaved_messages: 暂存的对话消息列表
    :param character_name: 当前 AI 角色的名称
    :param display_service: 演出规划器
    :return: 格式化后的对话历史字符串，被过滤的其他信息
    """
    chat_history = ""
    filtered_messages = []

    for msg in unsaved_messages:
        # 获取消息纯文本内容
        content = msg.content if hasattr(
            msg, 'content') else getattr(msg, 'text', '')

        if display_service is not None:
            content = display_service.parser_get_natural_text(content)

        # 1. 处理 AI 消息
        if isinstance(msg, AIMessage):
            if content and isinstance(content, str) and content.strip():
                chat_history += f"{character_name}: {content.strip()}\n"
        # 2. 处理 用户 消息
        elif isinstance(msg, HumanMessage):
            if content and isinstance(content, str) and content.strip():
                chat_history += f"User: {content.strip()}\n"
        
        # 3. 存储其他消息
        else:
            filtered_messages.append(msg)

    return chat_history, filtered_messages


def format_working_memory_for_prompt(
    working_memory_cache: List[Dict],
    include_timestamp: bool = True,
) -> str:
    """
    将移动缓存格式化为纯文本，供大模型直接读取。
    输出机制：严格按照时间戳先后顺序排列。
    """
    if not working_memory_cache:
        return "近期暂无工作记忆缓存。"

    # 按照时间戳重新排序（保证时间线顺畅）
    chronological_cache = sorted(
        working_memory_cache, key=lambda x: x['timestamp'])

    formatted_text = ""
    for mem in chronological_cache:
        details = f"重要性(1-10): {mem.get('importance', 0)}"
        if include_timestamp:
            details = (
                f"保存时间: {format_timestamp(mem.get('timestamp'))}, "
                f"{details}"
            )
        formatted_text += f"- [{details}] {mem.get('text')}\n"

    return formatted_text


def process_and_archive_memories(
    context_messages: list,
    unsaved_messages: list,
    character_name: str,
    character_setting_summary: str,
    memory_manager: object,
    llm: object,                     # 接收大模型实例
    memory_extraction_model: type,   # 接收 Pydantic 模型类
    true_character_name: str | None = None,
    display_service: CharacterDisplayService | None = None,
    scene_mode: str = 'realtime'
    # ignored_tools: list = None
) -> tuple[list[dict], Exception | str | None]:
    """
    核心记忆处理函数：格式化对话、提取记忆、入库，并更新移动缓存。

    大模型调用或解析失败时，最多尝试 2 次：
    - 第一次失败：重新调用一次大模型并重新解析
    - 第二次失败：返回最后一次异常

    返回：
        (提取后的记忆列表, 错误信息)
    """
    new_extracted_memories = []

    # 数据库存储使用稳定的目录名，提示词和对话文本使用真实角色名。
    memory_owner = character_name
    character_name = true_character_name or character_name

    if not unsaved_messages:
        return new_extracted_memories, '没有待保存消息！'

    print("\n[系统] 正在进行记忆整合与反思...")

    # 1. 格式化当前待保存对话
    unsaved_chat_history, _ = format_chat_history_only_human_ai(
        unsaved_messages=unsaved_messages,
        character_name=character_name,
        display_service=display_service
    )

    if not unsaved_chat_history.strip():
        return new_extracted_memories, '没有待保存消息！'

    # 2. 格式化上下文历史
    chat_history, _ = format_chat_history_only_human_ai(
        unsaved_messages=context_messages,
        character_name=character_name,
        display_service=display_service
    )

    if chat_history == '':
        chat_history = '暂无'

    system_prompt = ARCHIVE_MEMORY_PROMPTS[scene_mode].format(
        character_name=character_name,
        character_setting_summary=character_setting_summary,
    )

    # 3. 构造解析器和 Prompt
    parser = PydanticOutputParser(
        pydantic_object=memory_extraction_model
    )
    format_instructions = parser.get_format_instructions()

    final_system_prompt = (
        system_prompt
        + "\n\n"
        + format_instructions
    )

    human_prompt = (
        f"##对话记录历史\n\n"
        f"说明：提供“当前待总结对话”之前的对话内容，用于理解当前对话。"
        f"不要单独总结此部分的旧内容。\n\n"
        f"具体对话：\n{chat_history}\n\n"
        f"##当前待总结对话\n\n"
        f"说明：总结以下对话带来的新增信息或新形成的完整事件。"
        f"请参考对话记录历史理解上下文，但不要重复提取没有发生变化的旧信息。"
        f"如果当前对话包含对历史内容的回答、纠正、确认、补充或结果，"
        f"可以引用历史中最低限度的必要信息，使新记忆能够被独立理解。\n\n"
        f"具体对话：\n{unsaved_chat_history}"
    )

    print(
        f'\n[记忆整理, {character_name}] '
        f'待整理内容如下：\n{human_prompt}\n'
    )

    messages_for_extraction = [
        SystemMessage(content=final_system_prompt),
        HumanMessage(content=human_prompt)
    ]

    # 4. 大模型调用 + 解析，失败后重试一次
    result = None
    last_error = None
    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(
                    f'[记忆整理, {character_name}] '
                    f'第一次提取失败，正在进行第 {attempt} 次尝试...'
                )

            # 调用大模型
            response = llm.invoke(messages_for_extraction)

            # 解析大模型返回结果
            result = parser.parse(response.content)

            # 调用和解析均成功，退出重试循环
            break

        except Exception as e:
            last_error = e

            print(
                f'[记忆整理, {character_name}] '
                f'第 {attempt}/{max_attempts} 次提取失败：{e}'
            )

    # 两次都失败
    if result is None:
        print(
            f'[记忆整理, {character_name}] '
            f'连续 {max_attempts} 次提取失败，终止本次记忆整理。'
        )
        return new_extracted_memories, last_error

    # 5. 解析成功后再进行入库
    try:
        if result.memories:
            current_time = time.time()

            for mem in result.memories:
                memory_manager.add_memory(
                    memory_owner=memory_owner,
                    text=mem.text,
                    category=mem.category,
                    importance=mem.importance,
                    timestamp=current_time
                )

                print(
                    f"  -> [已归档记忆]: {mem.text} "
                    f"(类别: {mem.category}, 评级: {mem.importance})"
                )

                # 6. 组装缓存对象
                new_extracted_memories.append({
                    "text": mem.text,
                    "importance": mem.importance,
                    "timestamp": current_time
                })

        else:
            print(
                f'[记忆整理, {character_name}] '
                f'当前没有需要存储的记忆'
            )

        return new_extracted_memories, None

    except Exception as e:
        # 入库等后处理错误不触发大模型重试
        return new_extracted_memories, e


def update_working_memory_cache(
    working_memory_cache: List[Dict],
    working_memory_candidate: List[Dict],  # 存储未进入 cache 的记忆总结
    new_memories: List[Dict],              # 本轮记忆存储得到的记忆总结
    turn_range: Tuple[int, int],           # (start_turn, end_turn)
    max_cache_size: int = 20,              # 输入大模型的工作记忆数量上限
    base_message_turn: int = 20,            # 输入大模型的真实对话数量
    last_activated_turn: int = 0,
) -> Tuple[List[Dict], List[Dict], int]:
    """
    更新工作记忆缓存池，并管理记忆的“延迟激活”与“容量截断”。

    核心机制：
    采用“缓冲队列（Staging Buffer）”设计，彻底解决 LLM 上下文冗余（Context Overlap）问题。
    新总结的记忆会先进入候选池（candidate）暂存，通过比对对话轮数（end_turn），
    仅当这段记忆对应的原始对话【完全脱离】大模型的当前上下文窗口时，才会被激活并合入最终的工作记忆中。

    Args:
        working_memory_cache (List[Dict]): 
            当前正在生效的工作记忆缓存。里面存放的是已脱离原始对话窗口的旧历史总结。
        working_memory_candidate (List[Dict]): 
            记忆候选池（缓冲队列）。存放已总结完毕，但仍在等待被挤出原始上下文窗口的记忆。
        new_memories (List[Dict]): 
            本轮新提取出的记忆字典列表。
        turn_range (Tuple[int, int]): 
            本轮归档总结所对应的原始对话轮次范围，格式为 (起始轮数, 结束轮数)。
        max_cache_size (int, optional): 
            输入大模型的工作记忆数量上限。超出此数量时会根据时间戳剔除最旧的记忆。默认为 20。
        base_message_turn (int, optional): 
            大模型当前上下文实际保留的真实对话轮数。默认为 20。

    Returns:
        Tuple[List[Dict], List[Dict], int]:
            返回工作记忆、候选池和单调递增的最新激活轮次：
            - working_memory_cache: 更新并截断后的工作记忆池
            - working_memory_candidate: 剔除了已激活记忆后的候选池
            - working_memory_end_turn: 已退出原始窗口且完成记忆判断的最新轮次
    """
    if (
        len(turn_range) != 2
        or turn_range[0] <= 0
        or turn_range[1] < turn_range[0]
    ):
        raise ValueError(f"非法的记忆轮次范围：{turn_range}")
    if not isinstance(last_activated_turn, int) or last_activated_turn < 0:
        raise ValueError(
            f"非法的 last_activated_turn：{last_activated_turn!r}"
        )

    # 避免节点后续失败时，通过列表原地修改污染输入 state。
    working_memory_cache = list(working_memory_cache)
    working_memory_candidate = [
        {
            **candidate,
            "memories": list(candidate.get("memories", [])),
        }
        for candidate in working_memory_candidate
    ]

    # 成功完成记忆判断的批次即使没有提取出记忆，也要保留轮次水位，
    # 否则连续空结果会导致原始 messages 永远无法截断。
    working_memory_candidate.append({
        'memories': list(new_memories),
        'start_turn': turn_range[0],
        'end_turn': turn_range[1]
    })

    next_turn = turn_range[1] + 1  # 目标是为下一轮对话生成工作记忆

    # === 2. 计算阈值
    # 当前最新轮数 - 上下文保留轮数 = 已经滚出上下文窗口的最后轮数
    # 例如：下一轮是第 100 轮，窗口保留 20 轮（81~100）。那么结束轮次 <= 80 的记忆才算真正脱离了窗口。
    turn_to_be_cache = next_turn - base_message_turn
    if last_activated_turn > max(0, turn_to_be_cache):
        raise _message_state_error(
            f"last_activated_turn={last_activated_turn} 超过当前可激活水位 "
            f"{max(0, turn_to_be_cache)}"
        )

    retained_candidates = []
    memories_to_activate = []

    # === 3. 遍历候选池，筛选应该被激活的记忆
    working_memory_end_turn = last_activated_turn
    for cand in working_memory_candidate:
        # 使用 end_turn 判断，确保记忆对应的原始对话已完全离开上下文窗口
        if cand['end_turn'] <= turn_to_be_cache:
            # 提取出里面的多条记忆内容，准备加入主缓存
            memories_to_activate.extend(cand.get('memories', []))
            # 计算 working_memory_cache 中最新记录的轮数
            working_memory_end_turn = max(
                working_memory_end_turn, cand['end_turn']
            )
        else:
            # 尚未离开原始对话窗口，继续留在候选池中等待
            retained_candidates.append(cand)

    # 更新候选池
    working_memory_candidate = retained_candidates

    # === 4. 合并激活的记忆到主缓存
    if memories_to_activate:
        working_memory_cache.extend(memories_to_activate)

    # === 5. 排序与截断处理
    if working_memory_cache:

        if len(working_memory_cache) > max_cache_size:
            # 第一步：筛选淘汰 (优胜劣汰)
            # 按 importance（主）和 timestamp（次）进行降序排序
            # 这样最重要、且时间最新的记忆会排在列表最前面
            working_memory_cache.sort(
                key=lambda x: (x.get('importance', 0), x.get('timestamp', 0)),
                reverse=True
            )
            # 第二步：截断容量
            # 排在最前面的 max_cache_size 个核心记忆
            working_memory_cache = working_memory_cache[:max_cache_size]

        # 第三步：时间线重排 (整理队伍)
        # 将留下来的精英记忆按 timestamp 升序排序（由老到新）
        # 确保大模型阅读时，事件的发展顺序是符合时间逻辑的
        working_memory_cache.sort(key=lambda x: x.get('timestamp', 0))        

    return working_memory_cache, working_memory_candidate, working_memory_end_turn


def get_delete_instructions(state, message_len):
    current_messages = state["messages"]
    delete_instructions = []

    # 假设我们只保留最新的 20 条消息 (MESSAGE_LEN = 20)
    # 如果当前消息总数大于 20，我们就把前面的都删掉
    if len(current_messages) > message_len:
        messages_to_delete = current_messages[:-message_len]
        for msg in messages_to_delete:
            # 只有带有 id 的消息才能被删除（LangChain 较新版本默认会生成 id）
            if getattr(msg, "id", None):
                delete_instructions.append(RemoveMessage(id=msg.id))
            else:
                # 兼容处理：如果没有 id，说明版本较旧，无法精细删除
                pass

    return delete_instructions


# 旧版个人专辑（长期记忆）总结器，已经不再使用
def initialize_core_biography(
    character_name: str,
    character_setting_summary: str,
    memory_manager: object,
    llm: object,
    long_memory_len: int = 30,
    true_character_name: str | None = None,
    memory_root: str | Path | None = None,
    scene_mode: str = "realtime",
) -> str:
    """
    系统启动时调用：从数据库捞取顶级记忆，参考旧传记生成新传记。
    """
    if scene_mode not in VALID_SCENE_MODES:
        raise ValueError(f"非法 scene_mode：{scene_mode!r}")
    character_dir = Path(
        memory_root or f"./characters/{character_name}/memory_db"
    )
    display_name = true_character_name or character_name
    bio_path = character_dir / "core_memory.txt"

    # 1. 读取上一次的传记
    current_bio = "暂无。"
    if bio_path.exists():
        with bio_path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                current_bio = content

    # 2. 从数据库捞取前 N 条最核心的记忆
    # 注意：需确保你在 memory_db.py 中实现了类似 get_top_memories 的方法
    # 它的 SQL/查询逻辑必须是: ORDER BY importance DESC, timestamp DESC LIMIT {long_memory_len}
    try:
        top_memories = memory_manager.get_top_memories(limit=long_memory_len)
    except AttributeError:
        print(" [警告]: memory_manager 未实现 get_top_memories 方法，跳过传记重写。")
        return current_bio

    if not top_memories:
        return current_bio

    # 3. 在内存中按时间线重新理顺这些核心记忆
    top_memories_chronological = sorted(
        top_memories, key=lambda x: x.get('timestamp', 0))
    facts = []
    for mem in top_memories_chronological:
        details = f"重要性(1-10): {mem.get('importance')}"
        if scene_mode == "realtime":
            details = (
                f"保存时间: {format_timestamp(mem.get('timestamp'))}, "
                f"{details}"
            )
        facts.append(f"- [{details}] {mem.get('text')}")
    facts_text = "\n".join(facts)

    print("\n[系统初始化] 正在根据核心记忆库重写人物传记...")

    # 4. 构建重写 Prompt
    sys_prompt = f"""你是一位专业的人物传记作家。你的任务是维护AI伙伴【{display_name}】与用户【User】之间的关系传记（核心记忆）。

## AI伙伴的角色设定简述

{character_setting_summary}

## 上一次的传记记录

{current_bio}

## 当前数据库中最核心的 {long_memory_len} 条记忆事实（按时间先后排列）

{facts_text}

## 总结要求

请将这些核心事实与上一次的传记进行对比与融合。
要求：
1. 使用流畅的第三人称陈述句，专注于陈述事实，尽量少的加入自己的评价。
2. 提供角色设定简述是为了补充角色基础信息，请不要把角色设定直接复制或者改写到传记中。
3. 确保最新捞取的核心事实被合理地融入传记中。如果某事实已在旧传记中，请保留；如果旧传记中存在与新事实冲突的久远内容，以新事实为准进行改写。
4. 全文严格控制在 800 字以内。内容过多时请进行精简、或删除相对古老且重要程度低的记忆。
5. 直接输出新的传记文本，不要包含任何前缀或解释。"""

    try:
        response = llm.invoke([SystemMessage(content=sys_prompt)])
        new_bio = response.content.strip()

        # 5. 覆写本地文件，完成持久化
        bio_path.parent.mkdir(parents=True, exist_ok=True)
        with bio_path.open("w", encoding="utf-8") as f:
            f.write(new_bio)

        print(" [系统初始化] 传记重写完毕！")
        return new_bio
    except Exception as e:
        print(f" [传记重写出错] {e}")
        return current_bio


class MessageStateValidationError(RuntimeError):
    """消息轮次统计或工具调用协议已经不再可信。"""


def _message_state_error(reason: str) -> MessageStateValidationError:
    return MessageStateValidationError(
        f"消息状态校验失败：{reason}。"
        "建议立即停止当前对话并重新初始化角色，随后检查消息统计与截断逻辑。"
    )


def validate_tool_message_protocol(messages: list[BaseMessage]) -> None:
    """确保每条 ToolMessage 都属于前一条 AI 工具调用。"""
    pending_tool_call_ids: set[str] = set()

    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            if pending_tool_call_ids:
                raise _message_state_error(
                    f"第 {index} 条 HumanMessage 前仍缺少工具返回："
                    f"{sorted(pending_tool_call_ids)}"
                )
            continue

        if isinstance(message, AIMessage):
            if pending_tool_call_ids:
                raise _message_state_error(
                    f"第 {index} 条 AIMessage 前仍缺少工具返回："
                    f"{sorted(pending_tool_call_ids)}"
                )
            tool_calls = getattr(message, "tool_calls", None) or []
            tool_call_ids = [call.get("id") for call in tool_calls]
            if any(not call_id for call_id in tool_call_ids):
                raise _message_state_error(
                    f"第 {index} 条 AIMessage 包含没有 id 的工具调用"
                )
            if len(tool_call_ids) != len(set(tool_call_ids)):
                raise _message_state_error(
                    f"第 {index} 条 AIMessage 包含重复的工具调用 id"
                )
            pending_tool_call_ids = set(tool_call_ids)
            continue

        if isinstance(message, ToolMessage):
            tool_call_id = message.tool_call_id
            if tool_call_id not in pending_tool_call_ids:
                raise _message_state_error(
                    f"第 {index} 条 ToolMessage({tool_call_id}) "
                    "没有对应的前置 AI tool_calls"
                )
            pending_tool_call_ids.remove(tool_call_id)

    if pending_tool_call_ids:
        raise _message_state_error(
            f"消息末尾缺少工具返回：{sorted(pending_tool_call_ids)}"
        )


def update_num_msg_per_turn(state, this_turn: int) -> dict[int, int]:
    """计算本轮消息数并返回新的轮次统计，不直接修改输入 state。"""
    messages = list(state.get("messages", []))
    validate_tool_message_protocol(messages)

    num_msg_per_turn = dict(state.get("num_msg_per_turn", {}))
    for turn, count in num_msg_per_turn.items():
        if not isinstance(turn, int) or turn <= 0:
            raise _message_state_error(f"存在非法轮次键：{turn!r}")
        if not isinstance(count, int) or count <= 0:
            raise _message_state_error(
                f"第 {turn} 轮消息数非法：{count!r}"
            )
        if turn >= this_turn:
            raise _message_state_error(
                f"当前是第 {this_turn} 轮，但统计中已存在第 {turn} 轮"
            )

    recorded_turns = sorted(num_msg_per_turn)
    if recorded_turns:
        expected_turns = list(range(recorded_turns[0], this_turn))
        if recorded_turns != expected_turns:
            raise _message_state_error(
                "num_msg_per_turn 的轮次不连续；"
                f"实际为 {recorded_turns}，预期为 {expected_turns}"
            )

    prev_total_msgs = sum(num_msg_per_turn.values())
    if prev_total_msgs > len(messages):
        raise _message_state_error(
            "已有轮次统计总数 "
            f"{prev_total_msgs} 大于当前 messages 长度 {len(messages)}"
        )

    current_turn_msg_count = len(messages) - prev_total_msgs
    if current_turn_msg_count <= 0:
        raise _message_state_error(
            f"第 {this_turn} 轮没有可计入的新消息；"
            f"messages={len(messages)}，旧统计={prev_total_msgs}"
        )

    num_msg_per_turn[this_turn] = current_turn_msg_count
    if sum(num_msg_per_turn.values()) != len(messages):
        raise _message_state_error(
            "更新后的轮次统计总数与 messages 长度不一致"
        )
    return num_msg_per_turn


def truncate_messages(state, turn_start: int) -> dict:
    """按完整轮次生成 RemoveMessage 指令及截断后的轮次统计。"""
    messages = list(state.get("messages", []))
    num_msg_per_turn = dict(state.get("num_msg_per_turn", {}))

    if not isinstance(turn_start, int) or turn_start <= 0:
        raise _message_state_error(f"截断起始轮次非法：{turn_start!r}")
    if sum(num_msg_per_turn.values()) != len(messages):
        raise _message_state_error(
            "截断前 num_msg_per_turn 总数 "
            f"{sum(num_msg_per_turn.values())} 与 messages 长度 "
            f"{len(messages)} 不一致"
        )
    if num_msg_per_turn:
        latest_turn = max(num_msg_per_turn)
        if turn_start > latest_turn + 1:
            raise _message_state_error(
                f"截断起始轮次 {turn_start} 超过最新轮次 "
                f"{latest_turn} 的下一轮"
            )

    validate_tool_message_protocol(messages)
    deleted_counts = {
        turn: count
        for turn, count in num_msg_per_turn.items()
        if turn < turn_start
    }
    retained_counts = {
        turn: count
        for turn, count in num_msg_per_turn.items()
        if turn >= turn_start
    }
    delete_count = sum(deleted_counts.values())
    retained_messages = messages[delete_count:]

    if sum(retained_counts.values()) != len(retained_messages):
        raise _message_state_error(
            "按轮次计算的截断边界与实际 messages 长度不一致"
        )
    if retained_messages and not isinstance(retained_messages[0], HumanMessage):
        raise _message_state_error(
            "截断后的第一条消息不是 HumanMessage，边界落在了轮次内部"
        )
    validate_tool_message_protocol(retained_messages)

    delete_instructions = []
    for message in messages[:delete_count]:
        message_id = getattr(message, "id", None)
        if not message_id:
            raise _message_state_error(
                "待删除消息缺少 id，无法安全生成 RemoveMessage"
            )
        delete_instructions.append(RemoveMessage(id=message_id))

    result = {"num_msg_per_turn": retained_counts}
    if delete_instructions:
        result["messages"] = delete_instructions
    return result


_RECENT_REALTIME_PREFIX = re.compile(
    r"^\[\d{4}-\d{2}-\d{2}[^\]\r\n]*\]\s*"
)


class RecentMemoryManager:
    """管理近期记忆的暂存、总结与持久化。"""

    def __init__(
        self,
        character_name: str,
        memory_root: str | Path | None = None,
    ) -> None:
        character_name = character_name.strip()
        if not character_name:
            raise ValueError("character_name 不能为空")

        self.character_name = character_name
        root = Path(
            memory_root or Path("./characters") / character_name / "memory_db"
        )
        self.save_dir = root / "recent_memory"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.pending_memory_path = self.save_dir / "last_json_memory.json"
        self.recent_memory_path = self.save_dir / "recent_memory.txt"

    def save_new_extracted_memories(
        self,
        new_extracted_memories: Iterable[Any],
    ) -> None:
        """
        将新提取的记忆追加到 JSON 数组。
        输入格式为: [{"text": text,"importance": importance,"timestamp": current_time}, ...]
        """
        if not new_extracted_memories:
            return

        existing_records = self._read_json_memories()
        existing_records.extend(new_extracted_memories)
        self._write_json_memories(existing_records)

    def gen_recent_memory(
        self,
        llm: Any,
        max_json_items: int = 100,
        max_memories: int = 10,
        scene_mode: str = "realtime",
    ) -> str:
        """
        总结暂存记忆，并返回最近的若干行总结。

        Args:
            llm: 支持 invoke(messages) 方法的聊天模型。
            max_json_items: 单次最多交给模型处理的记忆数量。
            max_memories: 最多读取和返回的近期记忆行数。

        Returns:
            最近 max_memories 行近期记忆文本。
        """
        if scene_mode not in VALID_SCENE_MODES:
            raise ValueError(f"非法 scene_mode：{scene_mode!r}")

        if max_json_items <= 0:
            raise ValueError("max_json_items 必须大于 0")

        if max_memories <= 0:
            raise ValueError("max_memories 必须大于 0")

        new_json_memory = self._read_json_memories()

        # 只从文件尾部读取最近 max_memories 行。
        recent_memory = self._read_recent_memory(max_memories)

        # 没有新记忆时直接返回，不修改文件。
        if not new_json_memory:
            return self._for_scene(recent_memory, scene_mode)

        if len(new_json_memory) > max_json_items:
            # 先按重要性降序；重要性相同时，时间越新越靠前。
            new_json_memory = sorted(
                new_json_memory,
                key=lambda item: (
                    self._to_number(item.get("importance", 0)),
                    self._to_number(item.get("timestamp", 0)),
                ), reverse=True,
            )[:max_json_items]

        # 恢复为时间升序，帮助模型理解事件的发展过程。
        new_json_memory.sort(
            key=lambda item: self._to_number(item.get("timestamp", 0))
        )

        formatted_json_memory = "".join(
            (
                f"[保存时间: "
                f"{format_timestamp(item.get('timestamp', 0))}, "
                f"重要性(1-10): {item.get('importance', 0)}] "
                f"{item.get('text', '')}\n"
            )
            for item in new_json_memory
        ).rstrip()

        system_prompt = RECENT_MEMORY_PROMPTS[scene_mode].format(
            chat_records=formatted_json_memory
        )

        # 调用大模型。
        # 第一次失败后重试一次，总共最多调用两次。
        last_error = None
        last_memory = ""

        for attempt in range(2):
            try:
                response = llm.invoke([SystemMessage(content=system_prompt)])

                last_memory = getattr(response, "content", "")

                if not isinstance(last_memory, str) or not last_memory.strip():
                    raise RuntimeError("LLM 未返回有效的近期记忆总结")

                last_memory = last_memory.strip()
                if any(
                    line.strip()
                    and not _RECENT_REALTIME_PREFIX.match(line.strip())
                    for line in last_memory.splitlines()
                ):
                    raise RuntimeError(
                        "近期记忆总结存在缺少真实时间前缀的行"
                    )
                break

            except Exception as exc:
                last_error = exc

                # 第二次仍然失败，才真正向上抛出异常。
                if attempt == 1:
                    raise RuntimeError(
                        "LLM 调用失败，重试后仍未成功"
                    ) from last_error

        # 构造本次需要返回的近期记忆。
        combined_memory = "\n".join(
            part
            for part in (
                recent_memory.strip(),
                last_memory,
            )
            if part
        )

        recent_memory = self._keep_latest_lines(
            combined_memory,
            max_memories,
        )

        # 只追加本次生成的新记忆，不覆盖历史数据。
        self._append_recent_memory(last_memory)
        # recent_memory 成功写入后，再清空暂存记忆。
        self._write_json_memories([])

        return self._for_scene(recent_memory, scene_mode)

    @staticmethod
    def _for_scene(memory_text: str, scene_mode: str) -> str:
        """沙盒展示隐藏系统添加的现实时间，持久化文件保持不变。"""
        if not memory_text:
            return "暂无"
        if scene_mode == "realtime":
            return memory_text
        visible_lines = [
            _RECENT_REALTIME_PREFIX.sub("", line, count=1)
            for line in memory_text.splitlines()
        ]
        return "\n".join(visible_lines).strip() or "暂无"

    def _read_json_memories(self) -> list[dict[str, Any]]:
        """读取待总结的 JSON 记忆。"""
        if not self.pending_memory_path.exists():
            return []

        raw_content = self.pending_memory_path.read_text(
            encoding="utf-8"
        ).strip()

        if not raw_content:
            return []

        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"记忆文件不是有效 JSON："
                f"{self.pending_memory_path}"
            ) from exc

        if not isinstance(data, list):
            raise ValueError("记忆文件顶层结构必须是 JSON 数组")

        if not all(isinstance(item, dict) for item in data):
            raise ValueError("记忆文件中的每一项都必须是字典")

        return data

    def _write_json_memories(
        self,
        memories: list[dict[str, Any]],
    ) -> None:
        """以 JSON 数组格式保存待总结记忆。"""
        content = json.dumps(
            memories,
            ensure_ascii=False,
            indent=2,
        )
        self._atomic_write(self.pending_memory_path, content)
    
    def _read_recent_memory(self, max_memories: int) -> str:
        """读取最近 max_memories 行已经生成的近期记忆总结。"""
        if not self.recent_memory_path.exists():
            return ""

        with self.recent_memory_path.open("r", encoding="utf-8") as f:
            lines = deque(f, maxlen=max_memories)

        return "".join(lines).strip()

    def _write_recent_memory(self, content: str) -> None:
        """保存近期记忆，并统一文件末尾换行。"""
        file_content = f"{content}\n" if content else ""
        self._atomic_write(
            self.recent_memory_path,
            file_content,
        )

    def _append_recent_memory(self, content: str) -> None:
        """将新生成的近期记忆追加到文件末尾。"""
        content = content.strip()

        if not content:
            return

        self.recent_memory_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.recent_memory_path.open("a", encoding="utf-8") as f:
            f.write(content)
            f.write("\n")

    @staticmethod
    def _get_field(
        memory: Any,
        field: str,
        default: Any,
    ) -> Any:
        """同时兼容字典对象和普通属性对象。"""
        if isinstance(memory, Mapping):
            return memory.get(field, default)

        return getattr(memory, field, default)

    @staticmethod
    def _to_number(value: Any) -> float:
        """将时间和重要度转换为可排序的数值。"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _keep_latest_lines(
        content: str,
        max_lines: int,
    ) -> str:
        """移除空行并保留最后 max_lines 行。"""
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]
        return "\n".join(lines[-max_lines:])

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """
        先写入临时文件，再替换目标文件。

        可降低程序中途退出造成文件内容不完整的概率。
        """
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)


def load_json(path: Path) -> list[dict]:
    """读取 JSON 文件，便于测试断言。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_previous_turns(
    messages,
    unsaved_messages,
    num_msg_per_turn,
    turn_counter,
    n_turns=2,
):
    """
    从 messages 中获取 unsaved_messages 之前最多 n_turns 轮的对话。
    用于在总结记忆时提供上下文。

    特性：
    1. unsaved_messages 可以包含多轮。
    2. messages 中不足 n_turns 轮时，有多少返回多少。
    3. messages == unsaved_messages 时，上下文返回 []。
    4. 同时返回 unsaved_messages 对应的首个轮次。
    5. 两个消息列表、轮次统计或边界不一致时立即报错并建议停止对话。

    假设：
        正常情况下：
            messages = 部分历史消息 + unsaved_messages

        num_msg_per_turn:
            {
                1: 第1轮消息数,
                2: 第2轮消息数,
                ...
            }

        turn_counter:
            当前轮次。

    Returns:
        (context_messages, first_unsaved_turn)
    """

    if not unsaved_messages:
        raise _message_state_error("get_previous_turns 收到空的 unsaved_messages")
    if not messages:
        raise _message_state_error("存在待归档消息，但 messages 为空")
    if n_turns <= 0:
        return [], turn_counter

    # messages 中位于 unsaved_messages 之前，实际还有多少消息
    available_count = len(messages) - len(unsaved_messages)

    # messages 甚至没有 unsaved_messages 长，
    # 或者二者一样长，说明没有可用的历史上下文
    if available_count < 0:
        raise _message_state_error(
            f"messages 长度 {len(messages)} 小于 unsaved_messages 长度 "
            f"{len(unsaved_messages)}"
        )

    message_suffix = messages[available_count:]
    for offset, (message, unsaved_message) in enumerate(
        zip(message_suffix, unsaved_messages)
    ):
        message_id = getattr(message, "id", None)
        unsaved_id = getattr(unsaved_message, "id", None)
        if message_id and unsaved_id:
            matches = message_id == unsaved_id
        else:
            matches = message == unsaved_message
        if not matches:
            raise _message_state_error(
                "unsaved_messages 不是 messages 的完整尾部，"
                f"从第 {offset} 条待归档消息开始不一致"
            )

    if not isinstance(unsaved_messages[0], HumanMessage):
        raise _message_state_error(
            "unsaved_messages 第一条不是 HumanMessage，待归档范围落在轮次内部"
        )

    # --------------------------------------------------
    # 1. 根据 unsaved_messages 的长度，
    #    从当前轮向前推算 unsaved 从哪一轮开始
    # --------------------------------------------------

    unsaved_count = len(unsaved_messages)
    accumulated = 0
    first_unsaved_turn = turn_counter

    for turn in range(turn_counter, 0, -1):
        msg_count = num_msg_per_turn.get(turn)

        # 如果这一轮的统计不存在，就停止反推，
        # 使用目前能确定的信息
        if msg_count is None:
            raise _message_state_error(
                f"缺少第 {turn} 轮的 num_msg_per_turn 统计"
            )

        accumulated += msg_count
        first_unsaved_turn = turn

        if accumulated == unsaved_count:
            break
        if accumulated > unsaved_count:
            raise _message_state_error(
                "unsaved_messages 长度没有落在完整轮次边界上；"
                f"反推到第 {turn} 轮时累计 {accumulated} 条，"
                f"实际待归档 {unsaved_count} 条"
            )

    if accumulated != unsaved_count:
        raise _message_state_error(
            "无法使用 num_msg_per_turn 完整覆盖 unsaved_messages；"
            f"统计累计 {accumulated} 条，实际 {unsaved_count} 条"
        )

    # --------------------------------------------------
    # 2. 计算 unsaved 之前 n_turns 轮理论上的消息数量
    # --------------------------------------------------

    context_msg_count = 0

    for turn in range(
        first_unsaved_turn - 1,
        max(0, first_unsaved_turn - n_turns - 1),
        -1,
    ):
        context_msg_count += num_msg_per_turn.get(turn, 0)

    if context_msg_count <= 0:
        return [], first_unsaved_turn

    # --------------------------------------------------
    # 3. 实际可取数量不能超过 messages 当前拥有的历史数量
    # --------------------------------------------------

    actual_count = min(
        context_msg_count,
        available_count,
    )

    if actual_count <= 0:
        return [], first_unsaved_turn

    # unsaved_messages 正常情况下位于 messages 尾部
    unsaved_start = available_count

    return (
        messages[unsaved_start - actual_count:unsaved_start],
        first_unsaved_turn,
    )
