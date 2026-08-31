from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any
import json
import tomllib
import os
import re
from uuid import uuid4


DEFAULT_CONFIG_FILENAME = "character_config.default.toml"
CHARACTER_CONFIG_FILENAME = "character_config.toml"
FAVOURITE_GROUP_NAME = "喜爱"
UNGROUPED_GROUP_NAME = "未分组"
MAX_CHARACTER_GROUP_LENGTH = 30
VALID_SCENE_MODES = {"realtime", "sandbox"}
TRANSLATION_LANGUAGE_MAP = {
    "中文": "Chinese",
    "英语": "English",
    "德语": "German",
    "意大利语": "Italian",
    "葡萄牙语": "Portuguese",
    "西班牙语": "Spanish",
    "日语": "Japanese",
    "韩语": "Korean",
    "法语": "French",
    "俄语": "Russian",
}
DEFAULT_CHAT_THEME = "default"
VALID_CHAT_THEMES = {
    DEFAULT_CHAT_THEME,
    "rose_red",
    "sakura_pink",
    "royal_gold",
    "sky_blue",
    "sprout_green",
    "warm_orange",
    "mystic_purple",
    "midnight_black",
}
_CHAT_THEME_LINE = re.compile(r"^[ \t]*chat_theme[ \t]*=", re.MULTILINE)
_FAVOUR_LINE = re.compile(r"^[ \t]*favour[ \t]*=", re.MULTILINE)
_GROUP_LINE = re.compile(r"^[ \t]*group[ \t]*=", re.MULTILINE)
_CONFIG_WRITE_LOCK = Lock()


def _read_toml(path: Path) -> dict[str, Any]:
    """读取 TOML；配置错误应在角色初始化阶段直接暴露。"""
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"未找到默认角色配置文件：{path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"角色配置文件格式错误：{path}\n{error}") from error


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置，角色配置仅需填写想覆盖的项目。"""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def get_config_value(config: dict[str, Any], path: str) -> Any:
    """使用形如 memory.context.base_message_turn 的路径读取配置。"""
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"角色配置缺少必要项目：{path}")
        value = value[key]
    return value


def get_int_config(
    config: dict[str, Any],
    path: str,
    *,
    minimum: int = 1,
) -> int:
    value = get_config_value(config, path)
    # bool 是 int 的子类，但不应被当作数值配置接受。
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"角色配置 {path} 必须是不小于 {minimum} 的整数")
    return value


def get_string_list_config(config: dict[str, Any], path: str) -> list[str]:
    """读取由非空字符串组成的列表配置。"""
    value = get_config_value(config, path)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in value
    ):
        raise ValueError(f"角色配置 {path} 必须是非空字符串列表")
    return [item.strip() for item in value]


def normalize_translation_language(
    value: Any,
    path: str,
    *,
    allow_empty: bool = False,
) -> str:
    """将中英文语言配置规范为 TTS 接受的标准英文名称。"""
    if not isinstance(value, str):
        raise ValueError(f"角色配置 {path} 必须是字符串")

    language = value.strip()
    if not language:
        if allow_empty:
            return ""
        raise ValueError(f"角色配置 {path} 不能为空")

    if language in TRANSLATION_LANGUAGE_MAP:
        return TRANSLATION_LANGUAGE_MAP[language]
    if language in TRANSLATION_LANGUAGE_MAP.values():
        return language

    valid_languages = "、".join(
        [
            *TRANSLATION_LANGUAGE_MAP.keys(),
            *TRANSLATION_LANGUAGE_MAP.values(),
        ]
    )
    raise ValueError(
        f"角色配置 {path} 无效：{value!r}。有效值为：{valid_languages}"
    )


def get_memory_root(character_dir: Path, scene_mode: str) -> Path:
    """返回当前模式独立的记忆根目录。"""
    if scene_mode not in VALID_SCENE_MODES:
        raise ValueError(f"非法 scene_mode：{scene_mode!r}")
    directory_name = (
        "memory_db" if scene_mode == "realtime" else "memory_db_sandbox"
    )
    return character_dir / directory_name


def normalize_chat_theme(value: Any) -> str:
    """视觉配置错误不应阻止角色对话，未知值回退到简洁白。"""
    return (
        value
        if isinstance(value, str) and value in VALID_CHAT_THEMES
        else DEFAULT_CHAT_THEME
    )


def normalize_favour(value: Any) -> bool:
    """无效的界面喜爱配置按未喜爱处理，不阻止角色列表加载。"""
    return value if isinstance(value, bool) else False


def validate_character_group(value: Any, *, allow_empty: bool = True) -> str:
    """校验用户分组名；空字符串仅用于表示未指定自定义分组。"""
    if not isinstance(value, str):
        raise ValueError("group 必须是字符串")
    group = value.strip()
    if not group:
        if allow_empty:
            return ""
        raise ValueError("分组名称不能为空")
    if len(group) > MAX_CHARACTER_GROUP_LENGTH:
        raise ValueError(
            f"分组名称不能超过 {MAX_CHARACTER_GROUP_LENGTH} 个字符"
        )
    if group in {FAVOURITE_GROUP_NAME, UNGROUPED_GROUP_NAME}:
        raise ValueError(f"'{group}' 是系统保留分组名称")
    if any(ord(character) < 32 or ord(character) == 127 for character in group):
        raise ValueError("分组名称不能包含控制字符")
    return group


def normalize_character_group(value: Any) -> str:
    """手写分组配置无效时按未分组处理，避免阻断角色加载。"""
    try:
        return validate_character_group(value)
    except ValueError:
        return ""


def _replace_chat_theme(source: str, chat_theme: str) -> str:
    """只替换顶层 chat_theme 行，保留其余配置文本。"""
    lines = source.splitlines(keepends=True)
    section_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith("[")
        ),
        len(lines),
    )
    matches = [
        index
        for index, line in enumerate(lines[:section_index])
        if _CHAT_THEME_LINE.match(line)
    ]
    if len(matches) > 1:
        raise ValueError("角色配置存在重复的顶层 chat_theme")

    newline = "\r\n" if "\r\n" in source else "\n"
    theme_line = f'chat_theme = "{chat_theme}"'
    if matches:
        old_line = lines[matches[0]]
        line_ending = (
            "\r\n" if old_line.endswith("\r\n")
            else "\n" if old_line.endswith("\n")
            else ""
        )
        lines[matches[0]] = theme_line + line_ending
        return "".join(lines)

    if section_index == len(lines):
        if source and not source.endswith(("\n", "\r")):
            lines.append(newline)
        lines.append(theme_line + newline)
    else:
        lines.insert(section_index, theme_line + newline + newline)
    return "".join(lines)


def _replace_favour(source: str, favour: bool) -> str:
    """只替换顶层 favour 行，保留其余角色配置。"""
    lines = source.splitlines(keepends=True)
    section_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith("[")
        ),
        len(lines),
    )
    matches = [
        index
        for index, line in enumerate(lines[:section_index])
        if _FAVOUR_LINE.match(line)
    ]
    if len(matches) > 1:
        raise ValueError("角色配置存在重复的顶层 favour")

    newline = "\r\n" if "\r\n" in source else "\n"
    favour_line = f"favour = {'true' if favour else 'false'}"
    if matches:
        old_line = lines[matches[0]]
        line_ending = (
            "\r\n" if old_line.endswith("\r\n")
            else "\n" if old_line.endswith("\n")
            else ""
        )
        lines[matches[0]] = favour_line + line_ending
        return "".join(lines)

    if section_index == len(lines):
        if source and not source.endswith(("\n", "\r")):
            lines.append(newline)
        lines.append(favour_line + newline)
    else:
        lines.insert(section_index, favour_line + newline + newline)
    return "".join(lines)


def _replace_group(source: str, group: str) -> str:
    """只替换顶层 group 行，保留其余角色配置。"""
    lines = source.splitlines(keepends=True)
    section_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith("[")
        ),
        len(lines),
    )
    matches = [
        index
        for index, line in enumerate(lines[:section_index])
        if _GROUP_LINE.match(line)
    ]
    if len(matches) > 1:
        raise ValueError("角色配置存在重复的顶层 group")

    newline = "\r\n" if "\r\n" in source else "\n"
    group_line = f"group = {json.dumps(group, ensure_ascii=False)}"
    if matches:
        old_line = lines[matches[0]]
        line_ending = (
            "\r\n" if old_line.endswith("\r\n")
            else "\n" if old_line.endswith("\n")
            else ""
        )
        lines[matches[0]] = group_line + line_ending
        return "".join(lines)

    if section_index == len(lines):
        if source and not source.endswith(("\n", "\r")):
            lines.append(newline)
        lines.append(group_line + newline)
    else:
        lines.insert(section_index, group_line + newline + newline)
    return "".join(lines)


def save_character_chat_theme(
    characters_dir: Path,
    character_name: str,
    chat_theme: str,
) -> None:
    """原子保存唯一允许由主题接口修改的 chat_theme 配置。"""
    if chat_theme not in VALID_CHAT_THEMES:
        raise ValueError(f"不支持的聊天主题：{chat_theme}")

    characters_dir = characters_dir.resolve()
    character_dir = (characters_dir / character_name).resolve()
    if character_dir.parent != characters_dir or not character_dir.is_dir():
        raise ValueError(f"角色 '{character_name}' 不存在")

    default_path = characters_dir / DEFAULT_CONFIG_FILENAME
    target_path = character_dir / CHARACTER_CONFIG_FILENAME
    temporary_path = character_dir / f".{uuid4().hex}.toml.tmp"

    with _CONFIG_WRITE_LOCK:
        source_path = target_path if target_path.is_file() else default_path
        with source_path.open("r", encoding="utf-8", newline="") as file:
            source = file.read()
        updated = _replace_chat_theme(source, chat_theme)
        parsed = tomllib.loads(updated)
        if parsed.get("chat_theme") != chat_theme:
            raise ValueError("聊天主题配置写入校验失败")

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                file.write(updated)
            os.replace(temporary_path, target_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def save_character_favour(
    characters_dir: Path,
    character_name: str,
    favour: bool,
) -> None:
    """原子保存唯一允许由喜爱接口修改的 favour 配置。"""
    if not isinstance(favour, bool):
        raise ValueError("favour 必须是布尔值")

    characters_dir = characters_dir.resolve()
    character_dir = (characters_dir / character_name).resolve()
    if character_dir.parent != characters_dir or not character_dir.is_dir():
        raise ValueError(f"角色 '{character_name}' 不存在")

    target_path = character_dir / CHARACTER_CONFIG_FILENAME
    temporary_path = character_dir / f".{uuid4().hex}.toml.tmp"

    with _CONFIG_WRITE_LOCK:
        if target_path.is_file():
            with target_path.open("r", encoding="utf-8", newline="") as file:
                source = file.read()
        else:
            source = ""
        updated = _replace_favour(source, favour)
        parsed = tomllib.loads(updated)
        if parsed.get("favour") is not favour:
            raise ValueError("角色喜爱配置写入校验失败")

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                file.write(updated)
            os.replace(temporary_path, target_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def save_character_group(
    characters_dir: Path,
    character_name: str,
    group: str,
    *,
    clear_favour: bool = False,
) -> None:
    """原子保存角色分组；移至未分组时可同时取消喜爱。"""
    group = validate_character_group(group)

    characters_dir = characters_dir.resolve()
    character_dir = (characters_dir / character_name).resolve()
    if character_dir.parent != characters_dir or not character_dir.is_dir():
        raise ValueError(f"角色 '{character_name}' 不存在")

    target_path = character_dir / CHARACTER_CONFIG_FILENAME
    temporary_path = character_dir / f".{uuid4().hex}.toml.tmp"

    with _CONFIG_WRITE_LOCK:
        if target_path.is_file():
            with target_path.open("r", encoding="utf-8", newline="") as file:
                source = file.read()
        else:
            source = ""
        updated = _replace_group(source, group)
        if clear_favour:
            updated = _replace_favour(updated, False)
        parsed = tomllib.loads(updated)
        if parsed.get("group") != group:
            raise ValueError("角色分组配置写入校验失败")
        if clear_favour and parsed.get("favour") is not False:
            raise ValueError("角色喜爱配置写入校验失败")

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                file.write(updated)
            os.replace(temporary_path, target_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def load_character_config(
    characters_dir: Path,
    character_name: str,
) -> dict[str, Any]:
    """加载公共默认配置，再递归覆盖角色自己的可选配置。"""
    default_path = characters_dir / DEFAULT_CONFIG_FILENAME
    character_path = characters_dir / character_name / CHARACTER_CONFIG_FILENAME

    config = _read_toml(default_path)
    if character_path.is_file():
        config = merge_config(config, _read_toml(character_path))

    true_name = config.get("true_character_name")
    if not isinstance(true_name, str):
        raise ValueError("角色配置 true_character_name 必须是字符串")
    if not true_name.strip():
        config["true_character_name"] = character_name
    else:
        config["true_character_name"] = true_name.strip()

    scene_mode = config.get("scene_mode")
    if scene_mode not in VALID_SCENE_MODES:
        raise ValueError(
            "角色配置 scene_mode 只能是 'realtime' 或 'sandbox'"
        )

    config["chat_theme"] = normalize_chat_theme(config.get("chat_theme"))
    config["favour"] = normalize_favour(config.get("favour"))
    config["group"] = normalize_character_group(config.get("group", ""))

    # 在初始化时检查会共同影响容量清理的两个参数。
    max_capacity = get_int_config(
        config, "memory.retention.max_capacity"
    )
    safe_margin = get_int_config(
        config, "memory.retention.safe_margin", minimum=0
    )
    if max_capacity < 2 * safe_margin:
        raise ValueError(
            "角色配置 memory.retention.max_capacity "
            "必须不少于 safe_margin 的两倍"
        )

    config["display"]["tool_call_content_allowlist"] = (
        get_string_list_config(
            config,
            "display.tool_call_content_allowlist",
        )
    )

    return config
