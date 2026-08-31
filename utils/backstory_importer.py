"""角色背景故事文件的解析、校验与离线导入。"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


BACKSTORY_CATEGORY = "character_backstory"
BACKSTORY_IMPORTANCE = 7
BACKSTORY_TIMESTAMP = 0.0
MAX_BACKSTORY_CHARACTERS = 500
RECORD_SEPARATOR = "<<<RECORD_END>>>"


@dataclass(frozen=True)
class BackstoryFileSpec:
    filename: str
    id_prefix: str
    backstory_type: str
    required_header: str


BACKSTORY_FILE_SPECS = (
    BackstoryFileSpec(
        "01_核心历程.txt",
        "CoreJourney",
        "core_journey",
        "【事件】",
    ),
    BackstoryFileSpec(
        "02_补充经历.txt",
        "Extra_Experiences",
        "extra_experiences",
        "【经历】",
    ),
    BackstoryFileSpec(
        "03_人物印象.txt",
        "CharacterImpression",
        "character_impression",
        "【对象】",
    ),
    BackstoryFileSpec(
        "04_事物认知.txt",
        "EntityPerceptions",
        "entity_perceptions",
        "【对象】",
    ),
)

_SPEC_BY_PREFIX = {
    spec.id_prefix: spec for spec in BACKSTORY_FILE_SPECS
}
_BACKSTORY_ID_PATTERN = re.compile(
    r"^(CoreJourney|Extra_Experiences|CharacterImpression|EntityPerceptions)"
    r"([1-9]\d*)$"
)


class BackstoryImportError(RuntimeError):
    """背景故事无法安全导入。"""


class BackstoryValidationError(BackstoryImportError):
    """背景源文件未通过预检。"""


@dataclass(frozen=True)
class BackstoryRecord:
    text: str
    backstory_id: str
    backstory_type: str
    source_file: str
    record_index: int

    @property
    def first_line(self) -> str:
        return self.text.splitlines()[0]


@dataclass
class BackstoryImportReport:
    total_records: int
    added_records: int = 0
    deleted_records: int = 0
    records_by_file: dict[str, int] = field(default_factory=dict)


def backstory_metadata_from_id(backstory_id: str) -> dict[str, Any]:
    """解析并验证背景ID，返回可直接写入Chroma的元数据。"""
    match = _BACKSTORY_ID_PATTERN.fullmatch(backstory_id)
    if match is None:
        raise ValueError(f"非法 backstory_id：{backstory_id!r}")

    prefix, index_text = match.groups()
    spec = _SPEC_BY_PREFIX[prefix]
    return {
        "backstory_type": spec.backstory_type,
        "source_file": spec.filename,
        "record_index": int(index_text),
    }


def load_backstory_records(backstory_dir: str | Path) -> list[BackstoryRecord]:
    """读取并完整预检四个背景源文件，不执行任何数据库写入。"""
    source_dir = Path(backstory_dir)
    if not source_dir.is_dir():
        raise BackstoryValidationError(
            f"背景故事目录不存在：{source_dir}"
        )

    missing_files = [
        spec.filename
        for spec in BACKSTORY_FILE_SPECS
        if not (source_dir / spec.filename).is_file()
    ]
    if missing_files:
        missing_text = "、".join(missing_files)
        raise BackstoryValidationError(
            f"缺少必须的背景故事文件：{missing_text}"
        )

    records: list[BackstoryRecord] = []
    errors: list[str] = []

    for spec in BACKSTORY_FILE_SPECS:
        source_path = source_dir / spec.filename
        try:
            raw_text = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            errors.append(f"{spec.filename} 不是有效的UTF-8文件：{error}")
            continue

        if raw_text.strip() and not raw_text.rstrip().endswith(
            RECORD_SEPARATOR
        ):
            errors.append(
                f"{spec.filename} 的最后一条记录后缺少 "
                f"{RECORD_SEPARATOR}"
            )

        raw_records = [
            item.strip()
            for item in raw_text.split(RECORD_SEPARATOR)
            if item.strip()
        ]

        for record_index, text in enumerate(raw_records, start=1):
            first_line = text.splitlines()[0] if text.splitlines() else "(空记录)"
            if not first_line.startswith(spec.required_header):
                errors.append(
                    f"{spec.filename} 第{record_index}条的第一行格式错误："
                    f"{first_line}"
                )
            if len(text) > MAX_BACKSTORY_CHARACTERS:
                errors.append(
                    f"{spec.filename} 第{record_index}条超过"
                    f"{MAX_BACKSTORY_CHARACTERS}字符（实际{len(text)}字符）："
                    f"{first_line}"
                )

            backstory_id = f"{spec.id_prefix}{record_index}"
            records.append(
                BackstoryRecord(
                    text=text,
                    backstory_id=backstory_id,
                    backstory_type=spec.backstory_type,
                    source_file=spec.filename,
                    record_index=record_index,
                )
            )

    if errors:
        raise BackstoryValidationError(
            "背景故事预检失败，数据库未发生任何写入：\n- "
            + "\n- ".join(errors)
        )

    return records


def _get_existing_backstory(collection: Any) -> dict[str, Any]:
    """取得可用于失败回滚的完整背景故事快照。"""
    return collection.get(
        where={"category": BACKSTORY_CATEGORY},
        include=["documents", "metadatas", "embeddings"],
    )


def _delete_all_backstory(collection: Any) -> int:
    """删除collection中的全部背景故事，并返回删除条数。"""
    existing = collection.get(
        where={"category": BACKSTORY_CATEGORY},
        include=[],
    )
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)
    return len(existing_ids)


def _restore_backstory_snapshot(
    collection: Any,
    snapshot: dict[str, Any],
) -> None:
    """清除半成品并恢复导入前的全部背景故事。"""
    _delete_all_backstory(collection)
    old_ids = snapshot.get("ids", [])
    if not old_ids:
        return

    restore_args = {
        "ids": old_ids,
        "documents": snapshot.get("documents", []),
        "metadatas": snapshot.get("metadatas", []),
    }
    old_embeddings = snapshot.get("embeddings")
    if old_embeddings is not None:
        restore_args["embeddings"] = old_embeddings
    collection.add(**restore_args)


def import_backstory(
    memory_manager: Any,
    character_name: str,
    backstory_dir: str | Path,
    records: list[BackstoryRecord] | None = None,
) -> BackstoryImportReport:
    """预检通过后，全量替换角色collection中的背景故事。"""
    character_name = character_name.strip()
    if not character_name:
        raise BackstoryValidationError("角色名不能为空。")

    if records is None:
        records = load_backstory_records(backstory_dir)
    records_by_file = {
        spec.filename: sum(
            record.source_file == spec.filename for record in records
        )
        for spec in BACKSTORY_FILE_SPECS
    }
    report = BackstoryImportReport(
        total_records=len(records),
        records_by_file=records_by_file,
    )

    collection = memory_manager.collection
    snapshot = _get_existing_backstory(collection)

    try:
        report.deleted_records = _delete_all_backstory(collection)
        for record in records:
            memory_manager.add_memory(
                memory_owner=character_name,
                text=record.text,
                category=BACKSTORY_CATEGORY,
                importance=BACKSTORY_IMPORTANCE,
                timestamp=BACKSTORY_TIMESTAMP,
                backstory_id=record.backstory_id,
            )
            report.added_records += 1
    except Exception as error:
        try:
            _restore_backstory_snapshot(collection, snapshot)
        except Exception as rollback_error:
            raise BackstoryImportError(
                "背景故事全量更新失败，并且旧背景恢复失败："
                f"更新错误={error}；恢复错误={rollback_error}"
            ) from error
        raise BackstoryImportError(
            "背景故事全量更新失败；已清除本次半成品并恢复原有背景："
            f"{error}"
        ) from error

    return report
