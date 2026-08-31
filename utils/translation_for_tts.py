"""在 TTS 合成前翻译对话片段。"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.llm import get_translation_llm


logger = logging.getLogger(__name__)


class _TranslationBatch(BaseModel):
    """翻译模型应返回的结构化结果。"""

    translations: list[str] = Field(
        description="按原始顺序排列的对话翻译结果"
    )


class DialogueTranslationError(RuntimeError):
    """连续两次翻译均失败时抛出的异常。"""


class DialogueTranslatorForTTS:
    """通过一次模型调用，翻译一组用于 TTS 合成的对话片段。

    参数：
        target_language: 必填的目标语言，例如 ``"Japanese"``。
        source_language: 源语言，默认为 ``"Chinese"``。
        character_dir: 可选的角色目录，初始化时读取其中的
            ``translation_guide.txt`` 作为翻译指导；文件不存在或为空时忽略。
        llm: 可选的 LangChain 聊天模型。默认使用项目中低温度、关闭思考并
            限制等待时间的专用翻译模型。
        output_parser: 可选的、兼容 ``PydanticOutputParser`` 的解析器，

    ``translate`` 最多调用模型两次：首次调用失败后只重试一次。
    """

    def __init__(
        self,
        target_language: str,
        source_language: str = "Chinese",
        character_dir: str | Path | None = None,
        llm: Any | None = None,
        output_parser: Any | None = None,
    ) -> None:
        self.target_language = self._require_non_empty(
            target_language, "target_language"
        )
        self.source_language = self._require_non_empty(
            source_language, "source_language"
        )
        self.character_line_guide = self._load_translation_guide(character_dir)
        self.llm = llm or get_translation_llm()
        self.output_parser = output_parser or PydanticOutputParser(
            pydantic_object=_TranslationBatch
        )
        self.system_prompt = self._build_system_prompt()

    def translate(self, texts: Sequence[str]) -> list[str]:
        """翻译 ``texts``，并保持原有顺序和元素数量不变。

        输入为空时直接返回，不调用模型。模型输出无效、结果数量不符或请求
        失败时会重试一次；重试仍然失败则抛出 ``DialogueTranslationError``。
        """

        source_texts = self._validate_texts(texts)
        if not source_texts:
            return []

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return self._translate_once(source_texts)
            except Exception as exc:  # 模型服务或网络异常同样会触发重试。
                last_error = exc
                if attempt == 0:
                    logger.warning(
                        "对话翻译失败，将重试一次：%s",
                        exc,
                    )

        raise DialogueTranslationError(
            "对话翻译连续两次失败"
            f"({self.source_language} -> {self.target_language})."
        ) from last_error

    def __call__(self, texts: Sequence[str]) -> list[str]:
        """允许像调用函数一样直接调用翻译器实例。"""

        return self.translate(texts)

    def _translate_once(self, texts: list[str]) -> list[str]:
        request_text = json.dumps(
            {"texts": texts},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        response = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=request_text),
            ]
        )
        response_text = self._extract_response_text(response)
        parsed = self.output_parser.parse(response_text)
        translations = self._extract_translations(parsed)

        if len(translations) != len(texts):
            raise ValueError(
                "翻译结果数量不符："
                f"预期 {len(texts)} 条，实际收到 {len(translations)} 条。"
            )
        if any(
            not isinstance(item, str) or not item.strip()
            for item in translations
        ):
            raise TypeError("每一项翻译结果都必须是非空字符串。")

        return translations

    @staticmethod
    def _load_translation_guide(character_dir: str | Path | None) -> str:
        """仅在初始化时读取可选指导文件，读取失败不影响基础翻译。"""
        if character_dir is None:
            return ""

        guide_path = Path(character_dir) / "translation_guide.txt"
        try:
            return guide_path.read_text(encoding="utf-8-sig").strip()
        except FileNotFoundError:
            return ""
        except (OSError, UnicodeError) as exc:
            logger.warning(
                "无法读取角色翻译指导文件 %s，将使用无指导翻译：%s",
                guide_path,
                exc,
            )
            return ""

    def _build_system_prompt(self) -> str:
        prompt = (
            "## 翻译任务说明\n"
            f"将所给角色台词片段从{self.source_language}翻译为{self.target_language}。\n"
            "保持原意、语气、人名、顺序和条目数量，使用自然口语。\n"
        )
        if self.character_line_guide:
            language_style = (
                "### 角色语言风格说明\n"
                "以下内容可能包含角色说话习惯、口癖、标志性的台词、或者其他翻译要求，请参考其内容进行合理翻译：\n"
                f"{self.character_line_guide}"
            )
            prompt += language_style
        out_format = (
            "\n### 输出要求\n"
            '仅返回JSON：{"translations":["译文1","译文2"]}，不要解释。'
            '"translations"的长度要和输入台词片段的数量一致。'
        )
        prompt += out_format
        return prompt

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content

        # 部分兼容 OpenAI 的服务会返回内容块，而不是单个字符串；
        # JSON 解析器只需要其中的文本内容。
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
            if text_parts:
                return "".join(text_parts)

        raise TypeError("翻译模型没有返回文本内容。")

    @staticmethod
    def _extract_translations(parsed: Any) -> list[str]:
        if isinstance(parsed, _TranslationBatch):
            return parsed.translations
        if isinstance(parsed, dict) and isinstance(parsed.get("translations"), list):
            return parsed["translations"]
        translations = getattr(parsed, "translations", None)
        if isinstance(translations, list):
            return translations
        raise TypeError("输出解析器返回了不受支持的结果。")

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts 必须是字符串序列，不能是单个字符串。")

        result = list(texts)
        if any(not isinstance(item, str) for item in result):
            raise TypeError("每一项源文本都必须是字符串。")
        return result

    @staticmethod
    def _require_non_empty(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 必须是非空字符串。")
        return value.strip()
