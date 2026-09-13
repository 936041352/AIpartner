from collections import deque
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any
import json


class HistoryQueueEmptyError(LookupError):
    """当前没有可领取的历史事件。"""


class HistoryQueueOrderError(RuntimeError):
    """请求的历史事件不是当前 FIFO 队首。"""


class HistoryCursorError(ValueError):
    """分页游标不属于当前历史文件。"""


class ChatHistoryStore:
    """持久化角色对话，并向当前页面按 FIFO 交付新增事件。"""

    def __init__(self, memory_root: str | Path, max_turns: int = 1000):
        if max_turns <= 0:
            raise ValueError("max_turns 必须大于 0")

        self.max_turns = max_turns
        self.directory = Path(memory_root) / "chat_history"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "history.json"
        self._lock = Lock()
        self._ready_queue = deque()

    def start_turn(
        self,
        turn_id: str,
        user_text: str,
        speaker: str,
        created_at: float,
    ) -> dict[str, Any]:
        """创建新轮次并记录用户输入。"""
        event = self._new_event(
            turn_id=turn_id,
            sequence=1,
            event_type="user",
            role="user",
            speaker=speaker,
            content=user_text,
            created_at=created_at,
        )
        with self._lock:
            data = self._read_unlocked()
            if any(turn.get("turn_id") == turn_id for turn in data["turns"]):
                raise ValueError(f"历史轮次已存在：{turn_id}")

            # 轮次序号单调递增且永不复用：历史裁剪后旧目录仍能被准确
            # 判定为"已过期"，不会因为序号平移而错删仍在使用的语音。
            turn_number = int(data.get("next_turn_number", 1))
            data["next_turn_number"] = turn_number + 1

            data["turns"].append({
                "turn_id": turn_id,
                "turn_number": turn_number,
                "started_at": created_at,
                "completed_at": None,
                "status": "in_progress",
                "events": [event],
            })
            data["turns"] = data["turns"][-self.max_turns:]
            self._write_unlocked(data)
            self._ready_queue.append(deepcopy(event))
        return event

    def append_event(
        self,
        turn_id: str,
        event_type: str,
        role: str,
        speaker: str,
        content: str,
        created_at: float,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_names: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """将事件写入指定轮次；空内容不生成记录。"""
        content = content.strip()
        if not content:
            return None

        with self._lock:
            data = self._read_unlocked()
            turn = self._find_turn(data, turn_id)
            event = self._new_event(
                turn_id=turn_id,
                sequence=len(turn["events"]) + 1,
                event_type=event_type,
                role=role,
                speaker=speaker,
                content=content,
                created_at=created_at,
            )
            if tool_name:
                event["tool_name"] = tool_name
            if tool_call_id:
                event["tool_call_id"] = tool_call_id
            if tool_names:
                event["tool_names"] = list(tool_names)

            turn["events"].append(event)
            self._write_unlocked(data)
            self._ready_queue.append(deepcopy(event))
        return event

    def complete_turn(
        self,
        turn_id: str,
        speaker: str,
        content: str,
        completed_at: float,
    ) -> dict[str, Any]:
        """原子写入最终回复并将轮次标记为完成。"""
        with self._lock:
            data = self._read_unlocked()
            turn = self._find_turn(data, turn_id)
            event = self._new_event(
                turn_id=turn_id,
                sequence=len(turn["events"]) + 1,
                event_type="direct_reply",
                role="assistant",
                speaker=speaker,
                content=content,
                created_at=completed_at,
            )
            turn["events"].append(event)
            turn["completed_at"] = completed_at
            turn["status"] = "completed"
            self._write_unlocked(data)
            self._ready_queue.append(deepcopy(event))
        return event

    def fail_turn(self, turn_id: str, completed_at: float) -> None:
        """标记未能正常完成的轮次，不影响对话异常继续向上传递。"""
        with self._lock:
            data = self._read_unlocked()
            turn = self._find_turn(data, turn_id)
            turn["completed_at"] = completed_at
            turn["status"] = "failed"
            self._write_unlocked(data)

    def get_turns(
        self,
        limit: int = 10,
        before_turn_id: str | None = None,
    ) -> dict[str, Any]:
        """读取游标之前最近的若干完整轮次，返回顺序为旧到新。"""
        if limit <= 0:
            raise ValueError("limit 必须大于 0")

        with self._lock:
            data = self._read_unlocked()
            turns = data["turns"]
            end = len(turns)
            if before_turn_id is not None:
                for index, turn in enumerate(turns):
                    if turn.get("turn_id") == before_turn_id:
                        end = index
                        break
                else:
                    raise HistoryCursorError("历史分页游标不存在。")

            start = max(0, end - limit)
            items = deepcopy(turns[start:end])
            has_more = start > 0
            return {
                "turns": items,
                "has_more": has_more,
                "next_before_turn_id": (
                    items[0]["turn_id"] if has_more and items else None
                ),
                "total_turns": len(turns),
                "history_file_path": str(self.path.resolve()),
            }

    def snapshot(self) -> dict[str, str | None]:
        with self._lock:
            return {
                "ready_history_id": (
                    self._ready_queue[0]["event_id"]
                    if self._ready_queue
                    else None
                )
            }

    def pop_ready_event(self, event_id: str) -> dict[str, Any]:
        with self._lock:
            if not self._ready_queue:
                raise HistoryQueueEmptyError("当前没有可领取的历史事件。")
            event = self._ready_queue[0]
            if event["event_id"] != event_id:
                raise HistoryQueueOrderError(
                    "请求的历史事件不是当前待领取的队首。"
                )
            return self._ready_queue.popleft()

    def list_turn_ids(self) -> set[str]:
        """
        返回当前仍保留在历史中的全部轮次 ID。

        供语音文件清理使用：只有历史已丢弃的轮次，其音频才可以删除。
        """
        with self._lock:
            data = self._read_unlocked()
        turns = data.get("turns")
        if not isinstance(turns, list):
            return set()
        return {
            turn["turn_id"]
            for turn in turns
            if isinstance(turn, dict)
            and isinstance(turn.get("turn_id"), str)
        }

    def get_turn_number(self, turn_id: str) -> int | None:
        """
        返回轮次的稳定序号（从 1 开始，永不复用）。

        语音目录以该序号命名，因此必须与轮次一一对应且不随历史裁剪变化。
        旧版历史没有该字段时，退化为按当前位置推断。
        """
        with self._lock:
            data = self._read_unlocked()
        turns = data.get("turns")
        if not isinstance(turns, list):
            return None
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict) or turn.get("turn_id") != turn_id:
                continue
            number = turn.get("turn_number")
            if isinstance(number, int) and number >= 1:
                return number
            return index + 1
        return None

    def list_turn_numbers(self) -> set[int]:
        """
        返回历史中全部轮次的稳定序号。

        用于语音目录清理：历史只保留最近若干轮，序号不连续，必须按
        集合精确比对而不是按数量推断。
        """
        with self._lock:
            data = self._read_unlocked()
        turns = data.get("turns")
        if not isinstance(turns, list):
            return set()
        numbers: set[int] = set()
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            number = turn.get("turn_number")
            numbers.add(number if isinstance(number, int) and number >= 1 else index + 1)
        return numbers

    @staticmethod
    def _new_event(
        *,
        turn_id: str,
        sequence: int,
        event_type: str,
        role: str,
        speaker: str,
        content: str,
        created_at: float,
    ) -> dict[str, Any]:
        return {
            "turn_id": turn_id,
            "event_id": f"{turn_id}:{sequence:04d}",
            "type": event_type,
            "role": role,
            "speaker": speaker,
            "created_at": created_at,
            "content": content.strip(),
        }

    @staticmethod
    def _find_turn(data: dict[str, Any], turn_id: str) -> dict[str, Any]:
        for turn in reversed(data["turns"]):
            if turn.get("turn_id") == turn_id:
                return turn
        raise ValueError(f"历史轮次不存在：{turn_id}")

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema_version": 1,
                "max_turns": self.max_turns,
                "turns": [],
                "next_turn_number": 1,
            }

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"历史文件不是有效 JSON：{self.path}") from error

        if not isinstance(data, dict) or not isinstance(data.get("turns"), list):
            raise ValueError(f"历史文件结构错误：{self.path}")

        # 兼容旧文件：缺少计数器说明这是未迁移过的历史。此时把所有轮次按
        # 顺序统一重编号（旧轮次没有 turn_number 字段，只有新轮次才有，
        # 两者混用会出现"旧第 3 轮"和"新第 3 轮"抢同一个序号的冲突），
        # 并把计数器指向末尾之后。
        next_number = data.get("next_turn_number")
        if not isinstance(next_number, int) or next_number < 1:
            for index, turn in enumerate(data["turns"], start=1):
                if isinstance(turn, dict):
                    turn["turn_number"] = index
            data["next_turn_number"] = len(data["turns"]) + 1
        return data

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        data["schema_version"] = 1
        data["max_turns"] = self.max_turns
        content = json.dumps(data, ensure_ascii=False, indent=2)
        temporary_path = self.path.with_suffix(".json.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(self.path)
