"""把事件缓存全量同步到 Chroma；辅助同步失败只记录日志。"""

import json
import math
from pathlib import Path
from typing import Any


EPISODE_BUFFER_CATEGORY = "episode_buffer"


def sync_episode_buffer(
    memory_folder: str | Path, memory_manager: Any, character_name: str,
) -> int | None:
    """成功返回事件数（空缓存返回 0），失败打印错误并返回 None。

    JSON 是唯一数据源。先校验、向量化，再删除旧缓存并批量写入；
    不进行语义去重，不修改源文件，不重试或回滚。同一记忆目录须串行更新。
    """
    try:
        if not isinstance(character_name, str) or not character_name.strip():
            raise ValueError("角色名不能为空。")
        path = Path(memory_folder).expanduser() / "long_memory" / "core_episodes.json"
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)
        if not isinstance(state, dict) or not isinstance(state.get("episode_buffer"), list):
            raise ValueError("episode_buffer 必须是数组；缺失不能视为空缓存。")

        ids, documents, metadatas = [], [], []
        for episode in state["episode_buffer"]:
            if not isinstance(episode, dict) or any(
                not isinstance(episode.get(key), str) or not episode[key].strip()
                for key in ("id", "title", "content")
            ):
                raise ValueError("缓存事件缺少有效的 id、title 或 content。")
            importance = episode.get("importance")
            timestamp = episode.get("updated_at")
            status = episode.get("status")
            if (
                type(importance) is not int or not 1 <= importance <= 5
                or type(timestamp) not in (int, float)
                or not math.isfinite(timestamp) or timestamp < 0
                or status not in ("ongoing", "completed")
            ):
                raise ValueError("缓存事件的重要度、更新时间或状态无效。")
            episode_id = episode["id"].strip()
            ids.append(f"{EPISODE_BUFFER_CATEGORY}:{episode_id}")
            status_text = "进行中" if status == "ongoing" else "已完成"
            documents.append(
                f"{episode['title'].strip()}（{status_text}）\n{episode['content'].strip()}"
            )
            metadatas.append({
                "category": EPISODE_BUFFER_CATEGORY,
                "memory_owner": character_name.strip(),
                "episode_id": episode_id,
                "status": status,
                "importance": importance * 2,  # 对齐现有检索的 1～10 分尺度。
                "timestamp": float(timestamp),  # 维护时间，实际事件时间以正文为准。
            })
        if len(ids) != len(set(ids)):
            raise ValueError("缓存事件 ID 重复。")

        # 嵌入失败时保留旧缓存；写入时传入向量，避免删除后再次向量化。
        embeddings = memory_manager.embedding_fn(documents) if documents else None
        if documents and (embeddings is None or len(embeddings) != len(documents)):
            raise ValueError("缓存事件向量数量与正文数量不一致。")
        collection = memory_manager.collection
        # 类别和 ID 前缀均独立，普通记忆与背景故事不参与替换。
        collection.delete(where={"category": EPISODE_BUFFER_CATEGORY})
        if documents:
            collection.add(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings,
            )
        print(f"[事件缓存同步] 已同步 {len(ids)} 条事件。")
        return len(ids)
    except Exception as error:
        # 辅助功能不阻断聊天。删除后写入失败可能暂时缺失缓存，源 JSON 仍保留。
        print(f"[事件缓存同步失败] {error}")
        return None
