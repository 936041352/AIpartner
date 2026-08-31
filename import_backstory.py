"""把 characters/{角色名}/backstory 中的四类背景记录导入Chroma。"""

import argparse
from pathlib import Path
import sys

from utils.backstory_importer import (
    BackstoryImportError,
    import_backstory,
    load_backstory_records,
)
from utils.character_config import get_memory_root


COLLECTION_NAME = "partner_memory"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="预检并导入指定角色的背景故事。运行时请先关闭聊天程序。"
    )
    parser.add_argument("character_name", help="characters目录下的角色名")
    parser.add_argument(
        "--scene-mode",
        choices=("realtime", "sandbox"),
        default="realtime",
        help="写入哪一种模式的独立记忆库，默认 realtime",
    )
    return parser.parse_args()


def resolve_character_dir(project_root: Path, character_name: str) -> Path:
    characters_dir = (project_root / "characters").resolve()
    character_dir = (characters_dir / character_name.strip()).resolve()
    if (
        not character_name.strip()
        or character_dir.parent != characters_dir
        or not character_dir.is_dir()
    ):
        raise ValueError(f"角色目录不存在或角色名非法：{character_name!r}")
    return character_dir


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    try:
        character_dir = resolve_character_dir(
            project_root,
            args.character_name,
        )
        backstory_dir = character_dir / "backstory"
        # 必须先完成全部文件预检；失败时不创建数据库或collection。
        records = load_backstory_records(backstory_dir)
        print(
            f"[背景故事导入] 源文件预检完成，共 {len(records)} 条记录。",
            flush=True,
        )
        print(
            "[背景故事导入] 正在初始化Chroma和嵌入模型，请稍候……",
            flush=True,
        )
        # 延迟导入，让用户先看到预检进度；同时避免无效源文件触发模型加载。
        from utils.memory_db import MemoryManager

        memory_manager = MemoryManager(
            db_path=str(get_memory_root(character_dir, args.scene_mode)),
            collection_name=COLLECTION_NAME,
            embed_model_path=str(
                project_root / "weights" / "bge-base-zh-v1.5"
            ),
            scene_mode=args.scene_mode,
        )
        report = import_backstory(
            memory_manager=memory_manager,
            character_name=character_dir.name,
            backstory_dir=backstory_dir,
            records=records,
        )
    except (BackstoryImportError, ValueError) as error:
        print(f"[背景故事导入失败]\n{error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"[背景故事导入异常] {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print("\n=== 背景故事导入完成 ===")
    for filename, count in report.records_by_file.items():
        print(f"- {filename}: {count} 条")
    print(f"- 总计: {report.total_records} 条")
    print(f"- 已删除旧背景: {report.deleted_records} 条")
    print(f"- 新增: {report.added_records} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
