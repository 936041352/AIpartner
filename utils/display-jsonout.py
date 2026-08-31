from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

from typing import Literal
from pydantic import create_model, Field
from langchain_core.output_parsers import PydanticOutputParser

from .text_split import smart_split_text


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_DESCRIPTION = "请参考文件名称判断适用场景。"


@dataclass(frozen=True)
class PortraitAsset:
    id: str
    filename: str
    path: Path
    url: str
    description: str


@dataclass(frozen=True)
class VoiceAsset:
    id: str
    filename: str
    audio_path: Path       # 参考语义文件路径
    transcript_path: Path  # 参考文本文件路径
    transcript_text: str   # 参考文本
    description: str


class CharacterAssetCatalog:
    """扫描并缓存一个角色的立绘、参考语音及其说明。"""

    def __init__(self, project_root: Path, character_name: str):
        if not isinstance(project_root, Path):
            project_root = Path(project_root)

        self.character_name = character_name
        self.character_dir = project_root / "characters" / character_name
        self.portrait_dir = self.character_dir / "portraits"
        self.voice_dir = self.character_dir / "ref_audios"

        self.portraits = self._load_portraits()
        self.voices = self._load_voices()

        # list -> id 索引，方便后续快速查询
        self.portrait_by_id = {
            asset.id: asset
            for asset in self.portraits
        }
        self.voice_by_id = {
            asset.id: asset
            for asset in self.voices
        }
        self.default_portrait = self._select_default(self.portrait_by_id)
        self.default_voice = self._select_default(self.voice_by_id)

        self.portraits_prompt = self.prompt_catalog(self.portraits).strip()
        self.vioces_prompt = self.prompt_catalog(self.voices).strip()

        
    @staticmethod
    def _split_description_line(line: str) -> tuple[str, str] | None:
        """同时识别中英文冒号，并使用最靠前的冒号分隔。"""
        positions = [
            position
            for separator in (":", "：")
            if (position := line.find(separator)) >= 0
        ]
        if not positions:
            return None

        split_at = min(positions)
        filename = line[:split_at].strip()
        description = line[split_at + 1:].strip()
        if not filename:
            return None
        return filename, description

    def _read_descriptions(self, directory: Path) -> dict[str, str]:
        description_path = directory / "description.txt"
        if not description_path.is_file():
            print(f"[资源说明] 未找到 {description_path}，将使用默认说明。")
            return {}

        descriptions: dict[str, str] = {}
        for line_number, raw_line in enumerate(
            description_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue
            parsed = self._split_description_line(line)
            if parsed is None:
                print(f"[资源说明警告] {description_path}:{line_number} 无法解析，已忽略。")
                continue
            filename, description = parsed
            descriptions[filename] = description or DEFAULT_DESCRIPTION
        return descriptions

    @staticmethod
    def _ensure_unique_ids(assets: Sequence, resource_name: str) -> None:
        seen: set[str] = set()
        for asset in assets:
            normalized_id = asset.id
            if normalized_id in seen:
                raise ValueError(f"{resource_name}存在重复资源 ID：{asset.id}")
            seen.add(normalized_id)

    @staticmethod
    def _warn_unused_descriptions(
        descriptions: dict[str, str],
        actual_filenames: set[str],
    ) -> None:
        for filename in sorted(set(descriptions) - actual_filenames):
            print(f"[资源说明警告] {filename} 对应的资源不存在，修正文件中将忽略该条目。")

    @staticmethod
    def _write_rectified_description(
        directory: Path,
        entries: list[tuple[str, str]],
        resource_name: str,
    ) -> None:
        content = "\n".join(
            f"{filename}: {description}"
            for filename, description in entries
        )
        if content:
            content += "\n"

        output_path = directory / "description_rectified.txt"
        output_path.write_text(content, encoding="utf-8")
        print(f"\n========== {resource_name}说明修正结果 ==========")
        print(content.rstrip() or "（无有效资源）")
        print(f"输出文件：{output_path}")

    def _load_portraits(self) -> dict[str, PortraitAsset]:
        if not self.portrait_dir.is_dir():
            raise ValueError(f"立绘目录不存在：{self.portrait_dir}")

        # {'default.png': '用于通用的一般对话。', 'happy.png': '用于被夸奖、收到礼物或日常心情愉快时。'}
        descriptions = self._read_descriptions(self.portrait_dir)

        # 实际立绘图片路径
        # 排序时，将名称为default的文件放在最开头
        paths = sorted(
            (
                path
                for path in self.portrait_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            ),
            key=lambda path: (path.stem.casefold() != "default", path.name),
        )
        if not paths:
            raise ValueError(f"没有找到有效立绘：{self.portrait_dir}")

        actual_filenames = {path.name for path in paths}
        self._warn_unused_descriptions(descriptions, actual_filenames)
        assets = [
            PortraitAsset(
                id=f'portrait_{path.stem}',  # ID
                filename=path.name,  # 文件名称
                path=path,           # 文件路径
                url=(
                    f"/api/characters/"
                    f"{quote(self.character_name, safe='')}/portraits/"
                    f"{quote(path.name, safe='')}"
                ),
                description=descriptions.get(
                    path.name, DEFAULT_DESCRIPTION),  # 文件应用描述
            )
            for path in paths
        ]
        self._ensure_unique_ids(assets, "立绘")
        self._write_rectified_description(
            self.portrait_dir,
            [(asset.filename, asset.description) for asset in assets],
            "立绘",
        )
        return assets

    def _load_voices(self) -> dict[str, VoiceAsset]:
        if not self.voice_dir.is_dir():
            raise ValueError(f"参考语音目录不存在：{self.voice_dir}")

        # {'default.wav': '用于通用的一般对话。'}
        descriptions = self._read_descriptions(self.voice_dir)

        # 获取所有参考语音的文本
        transcript_paths = {
            path.stem: path
            for path in self.voice_dir.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".txt"
            and path.name
            not in {"description.txt", "description_rectified.txt"}
        }

        assets: list[VoiceAsset] = []

        # 实际参考语音路径
        # 排序时，将名称为default的文件放在最开头
        audio_paths = sorted(
            (
                path
                for path in self.voice_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in [".wav", ".mp3"]
            ),
            key=lambda path: (path.stem.casefold() != "default", path.name),
        )
        for audio_path in audio_paths:
            transcript_path = transcript_paths.get(audio_path.stem)

            # 忽略没有文本注释的参考语音
            if transcript_path is None:
                print(f"[资源说明警告] {audio_path.name} 缺少同名 TXT，已忽略。")
                continue

            assets.append(
                VoiceAsset(
                    id=f'voice_{audio_path.stem}',
                    filename=audio_path.name,
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    transcript_text=transcript_path.read_text(
                        encoding="utf-8").strip(),
                    description=descriptions.get(
                        audio_path.name, DEFAULT_DESCRIPTION)
                )
            )

        if not assets:
            raise ValueError(
                f"没有找到同时具备 WAV/MP3 和同名 TXT 的参考语音：{self.voice_dir}"
            )

        actual_filenames = {asset.filename for asset in assets}
        self._warn_unused_descriptions(descriptions, actual_filenames)
        self._ensure_unique_ids(assets, "参考语音")
        self._write_rectified_description(
            self.voice_dir,
            [(asset.filename, asset.description) for asset in assets],
            "参考语音",
        )
        return assets

    def _select_default(self, assets_by_id: dict):
        preferred_names = (
            "default", f"{self.character_name}_default".casefold(),
        )
        filename_to_id = {
            Path(asset.filename).stem.casefold(): asset_id
            for asset_id, asset in assets_by_id.items()
        }

        for preferred_name in preferred_names:
            actual_id = filename_to_id.get(preferred_name)
            if actual_id is not None:
                return assets_by_id[actual_id]

        first_id = next(iter(assets_by_id))
        print(f"[资源说明警告] 未找到 default 资源，将使用：{first_id}")
        return assets_by_id[first_id]

    @staticmethod
    def prompt_catalog(assets) -> str:
        """向规划模型提供的提示词。"""
        return ''.join(
            f'- [{asset.id}] {asset.description}\n'
            for asset in assets
        )


class CharacterDisplayService:
    def __init__(
        self,
        project_root: Path,
        character_name: str,
        true_character_name: str | None = None,
    ):
        self.catalog = CharacterAssetCatalog(project_root, character_name)
        # 资源路径仍使用 character_name，演出提示词使用面向用户的名称。
        self.true_character_name = true_character_name or character_name
        self.last_choosen = {
            'portrait_id': self.catalog.default_portrait.id,
            'voice_id': self.catalog.default_voice.id,
            'error': False
        }

        self.parser = self.get_output_parser()
        self.format_instructions = self.parser.get_format_instructions()

        self.only_default = len(self.catalog.portraits) == 1 and len(self.catalog.voices) == 1

    @property
    def default_portrait_url(self) -> str:
        """
        返回关闭演出规划时使用的默认立绘 URL。
        """
        return self.catalog.default_portrait.url

    def get_output_parser(self) -> PydanticOutputParser:
        """
        根据当前角色实际拥有的资源动态创建输出 Parser。
        portrait_id / voice_id 使用 Literal，因此模型输出的资源 ID 必须严格属于当前已有资源。
        """

        portrait_ids = tuple(asset.id for asset in self.catalog.portraits)
        voice_ids = tuple(asset.id for asset in self.catalog.voices)

        if not portrait_ids:
            raise ValueError(
                "[CharacterDisplayService.get_output_parser] "
                "当前角色没有可用的立绘资源。"
            )

        if not voice_ids:
            raise ValueError(
                "[CharacterDisplayService.get_output_parser] "
                "当前角色没有可用的语音资源。"
            )

        # 动态构造：
        # Literal["portrait_default", "portrait_happy", ...]
        # Literal["voice_default", "voice_happy", ...]
        PortraitId = Literal.__getitem__(portrait_ids)
        VoiceId = Literal.__getitem__(voice_ids)

        DynamicResourceAssignment = create_model(
            "ResourceAssignment",
            reply_section=(
                str,
                Field(
                    description=(
                        "此段回复给用户的文本。"
                        "根据演出需要决定分段长度，"
                        "建议每段至少 5 个字。"
                    )
                ),
            ),
            portrait_id=(
                PortraitId,
                Field(
                    description=(
                        "本段文本使用的立绘资源 ID。"
                        "必须从当前角色已有的立绘资源中选择。"
                    )
                ),
            ),
            voice_id=(
                VoiceId,
                Field(
                    description=(
                        "本段文本使用的参考语音资源 ID。"
                        "必须从当前角色已有的语音资源中选择。"
                    )
                ),
            ),
        )

        DynamicPlannerResponse = create_model(
            "PlannerResponse",
            assignments=(
                list[DynamicResourceAssignment],
                Field(
                    description=("按照先后顺序排列的回复文本分段，及其选择的立绘资源和语音资源")
                ),
            ),
        )

        return PydanticOutputParser(
            pydantic_object=DynamicPlannerResponse
        )
    
    def _plan_with_parser(self, ai_reply: str) -> dict[str, list]:
        """
        解析大模型已经生成好的演出规划。

        输入示例：

        {
            "assignments": [
                {
                    "reply_section": "今天真开心！我们出去玩吧。",
                    "portrait_id": "portrait_happy",
                    "voice_id": "voice_happy"
                },
                {
                    "reply_section": "不过晚上要早点回来。",
                    "portrait_id": "portrait_serious",
                    "voice_id": "voice_default"
                }
            ]
        }
        """

        # --------------------------------------------------
        # 1. 解析结构化演出回复
        # --------------------------------------------------

        try:
            result = self.parser.parse(ai_reply)

        except Exception as exc:
            self.last_choosen["error"] = True

            raise ValueError(
                f"[CharacterDisplayService.plan, {self.true_character_name}] "
                "无法解析大模型返回的演出规划。"
            ) from exc

        # --------------------------------------------------
        # 2. 初始化最终结果
        # --------------------------------------------------

        reply_chunks: list[str] = []
        character_ref_audios: list = []
        character_ref_texts: list[str] = []
        character_portraits: list[str] = []

        # --------------------------------------------------
        # 3. 获取 fallback 资源
        # --------------------------------------------------

        current_portrait_id = self.last_choosen["portrait_id"]
        current_voice_id = self.last_choosen["voice_id"]

        has_resource_error = False

        # --------------------------------------------------
        # 4. 逐个处理 assignment
        # --------------------------------------------------

        for assignment in result.assignments:

            # ------------------------------
            # 4.1 回复文本
            # ------------------------------

            reply_section = (assignment.reply_section.strip())

            # 空文本没有展示意义，忽略。
            if not reply_section:
                continue

            # ------------------------------
            # 4.2 立绘资源
            # ------------------------------

            requested_portrait_id = assignment.portrait_id.strip()

            if requested_portrait_id in self.catalog.portrait_by_id.keys():
                portrait_id = requested_portrait_id
            else:
                # 沿用上一段有效资源
                portrait_id = current_portrait_id
                has_resource_error = True

                print(
                    f"[CharacterDisplayService.plan, {self.true_character_name}] "
                    "不存在立绘资源 "
                    f"{requested_portrait_id!r}，"
                    f"沿用 {portrait_id!r}。"
                )

            # ------------------------------
            # 4.3 语音资源
            # ------------------------------

            requested_voice_id = assignment.voice_id.strip()

            if requested_voice_id in self.catalog.voice_by_id.keys():
                voice_id = requested_voice_id
            else:
                # 沿用上一段有效资源
                voice_id = current_voice_id
                has_resource_error = True

                print(
                    f"[CharacterDisplayService.plan, {self.true_character_name}] "
                    "不存在语音资源 "
                    f"{requested_voice_id!r}，"
                    f"沿用 {voice_id!r}。"
                )

            # ------------------------------
            # 4.4 获取真正的 Asset
            # ------------------------------

            portrait = self.catalog.portrait_by_id[portrait_id]
            voice = self.catalog.voice_by_id[voice_id]

            # ------------------------------
            # 4.5 二次切分文本
            # ------------------------------

            section_chunks = smart_split_text(reply_section)
            chunk_count = len(section_chunks)

            # ------------------------------
            # 4.6 展开文本
            # ------------------------------

            reply_chunks.extend(section_chunks)

            # ------------------------------
            # 4.7 展开立绘
            # ------------------------------

            character_portraits.extend([portrait.url] * chunk_count)

            # ------------------------------
            # 4.8 展开参考语音
            # ------------------------------

            character_ref_audios.extend([voice.audio_path] * chunk_count)
            character_ref_texts.extend([voice.transcript_text] * chunk_count)

            # ------------------------------
            # 4.9 更新当前有效资源
            # ------------------------------

            # 下一 assignment 如果输出非法 ID，
            # 就会沿用这里的资源。
            current_portrait_id = portrait_id
            current_voice_id = voice_id

        # --------------------------------------------------
        # 5. 检查是否实际获得回复
        # --------------------------------------------------

        if not reply_chunks:
            self.last_choosen["error"] = True

            raise ValueError(
                f"[CharacterDisplayService.plan, {self.true_character_name}] "
                "演出规划中未获得任何有效回复文本。"
            )

        # --------------------------------------------------
        # 8. 更新上一次选择
        # --------------------------------------------------

        self.last_choosen["portrait_id"] = current_portrait_id
        self.last_choosen["voice_id"] = current_voice_id
        self.last_choosen["error"] = has_resource_error

        # --------------------------------------------------
        # 8. 输出调试信息
        # --------------------------------------------------

        print(
            "\n"
            f"[CharacterDisplayService.plan, {self.true_character_name}] "
            "演出规划解析结果如下：\n"
        )

        for (chunk, portrait_url, audio_path,) in zip(
            reply_chunks,
            character_portraits,
            character_ref_audios,
        ):
            print(
                f"- [{chunk}]\n"
                f"- [{portrait_url}]\n"
                f"- [{audio_path}]"
            )

        # --------------------------------------------------
        # 9. 返回
        # --------------------------------------------------

        return {
            "reply_chunks": reply_chunks,
            "character_ref_audios": character_ref_audios,
            "character_ref_texts": character_ref_texts,
            "character_portraits": character_portraits,
        }

    def _plan_with_default(self, ai_reply: str) -> dict[str, list]:
        """
        当只有一个立绘和参考语音时使用
        """
        # 对AI回复进行分块
        chunks = smart_split_text(ai_reply)
        if not chunks:
            raise ValueError(f'[CharacterDisplayService.plan, {self.true_character_name}] 未获得有效回复！')
        choosen_portrait_urls = [
            self.catalog.default_portrait.url] * len(chunks)
        character_ref_audios = [
            self.catalog.default_voice.audio_path] * len(chunks)
        character_ref_texts = [
            self.catalog.default_voice.transcript_text] * len(chunks)

        return {
            "reply_chunks": chunks,
            "character_ref_audios": character_ref_audios,
            "character_ref_texts": character_ref_texts,
            "character_portraits": choosen_portrait_urls,
        }

    def plan(self, ai_reply: str) -> dict[str, list]:
        if self.only_default:
            return self._plan_with_default(ai_reply)
        else:
            return self._plan_with_parser(ai_reply)

    def parser_get_natural_text(
        self,
        ai_reply: str,
        join_sign: str = ""
    ) -> str:
        """
        将结构化演出回复转换成普通自然语言回复。
        """

        try:
            result = self.parser.parse(ai_reply)

        except Exception as exc:
            raise ValueError(
                "[CharacterDisplayService."
                "parser_get_natural_text] "
                "无法解析大模型返回的演出规划。"
            ) from exc

        sections = []

        for assignment in result.assignments:
            text = assignment.reply_section.strip()

            if text:
                sections.append(text)

        return join_sign.join(sections)


if __name__ == '__main__':
    project_root = Path('')
    character_name = 'sisi'
    display_service = CharacterDisplayService(
        project_root=project_root,
        character_name=character_name,
    )
    ai_reply = """
    {
        "assignments": [
            {
                "reply_section": "第一段，轻轻的我走了，正如我轻轻的来，我挥一挥衣袖，不带走一片云彩。",
                "portrait_id": "portrait_happy",
                "voice_id": "voice_happy"
            },
            {
                "reply_section": "第二段",
                "portrait_id": "portrait_sad",
                "voice_id": "voice_default"
            }
        ]
    }
    """
    results = display_service.plan(ai_reply)
    print(results)
    # print(results)
