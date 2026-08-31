import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .character_config import validate_character_group


GROUPS_RECORD_FILENAME = "groups_record.json"


class CharacterGroupRecordError(ValueError):
    """分组顺序记录损坏或格式不合法。"""


class CharacterGroupStore:
    """只保存当前非空自定义分组的显示顺序。"""

    def __init__(self, characters_dir: Path):
        self.characters_dir = characters_dir.resolve()
        self.record_path = self.characters_dir / GROUPS_RECORD_FILENAME
        self._lock = RLock()

    def _read(self) -> list[str]:
        if not self.record_path.is_file():
            return []
        try:
            data = json.loads(self.record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CharacterGroupRecordError("分组顺序记录读取失败") from error

        groups = data.get("groups") if isinstance(data, dict) else None
        if not isinstance(groups, list):
            raise CharacterGroupRecordError("分组顺序记录格式错误")

        normalized = []
        try:
            for group in groups:
                group = validate_character_group(group, allow_empty=False)
                if group not in normalized:
                    normalized.append(group)
        except ValueError as error:
            raise CharacterGroupRecordError("分组顺序记录格式错误") from error
        return normalized

    def _write(self, groups: list[str]) -> None:
        temporary_path = (
            self.characters_dir / f".{uuid4().hex}.groups.tmp"
        )
        content = json.dumps(
            {"version": 1, "groups": groups},
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        try:
            temporary_path.write_text(content, encoding="utf-8")
            os.replace(temporary_path, self.record_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _normalize_referenced(groups: list[str]) -> list[str]:
        normalized = []
        for group in groups:
            group = validate_character_group(group, allow_empty=False)
            if group not in normalized:
                normalized.append(group)
        return normalized

    def reconcile(self, referenced_groups: list[str]) -> list[str]:
        """保留仍非空的既有顺序，并将手写的新分组追加到末尾。"""
        referenced = self._normalize_referenced(referenced_groups)
        referenced_set = set(referenced)
        with self._lock:
            recorded = self._read()
            ordered = [group for group in recorded if group in referenced_set]
            ordered.extend(group for group in referenced if group not in ordered)
            if ordered != recorded:
                self._write(ordered)
            return ordered

    def move(
        self,
        group: str,
        direction: str,
        referenced_groups: list[str],
    ) -> list[str]:
        group = validate_character_group(group, allow_empty=False)
        if direction not in {"up", "down"}:
            raise ValueError("分组移动方向只能是 up 或 down")

        with self._lock:
            ordered = self.reconcile(referenced_groups)
            if group not in ordered:
                raise ValueError(f"分组 '{group}' 不存在")
            index = ordered.index(group)
            target = index - 1 if direction == "up" else index + 1
            if 0 <= target < len(ordered):
                ordered[index], ordered[target] = ordered[target], ordered[index]
                self._write(ordered)
            return ordered
